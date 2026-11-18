"""Socratic, curriculum-grounded tutor. The LLM backend is swappable."""
from __future__ import annotations
import logging
import re
from typing import Protocol
from .faithfulness import check
from .models import TutorReply
from .retriever import BM25Index, tokenize

# The guiding question (hint) is made separately; the model only writes the answer shown after the student asks.
SOCRATIC_PROMPT = """You are a patient science tutor for a {level} student.
Answer ONLY from the textbook excerpts below. If they do not contain the answer, say you cannot find it.
Write 2 to 4 short, simple sentences in {lang}. Do not add facts that are not in the excerpts.
Local examples (teff, injera, Ethiopian birr) are fine only if they do not change the facts.

Textbook excerpts:
{context}

Student question: {question}
"""


log = logging.getLogger("aitutor")


class ModelUnavailable(Exception):
    """The language model cannot be used (engine not installed, file missing, or it failed to load)."""


class Backend(Protocol):
    def generate(self, prompt: str) -> str: ...


class ExtractiveBackend:
    """No-model fallback: returns the best-matching sentence. Always available offline."""

    def best_sentence(self, question: str, text: str) -> str:
        q = set(tokenize(question))
        sents = re.split(r"(?<=[.!?])\s+", text)
        return max(sents, key=lambda s: len(q & set(tokenize(s))))


class LlamaCppBackend:
    """On-device LLM via llama.cpp (GGUF, INT4). Install: pip install llama-cpp-python"""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int | None = None,
                 n_gpu_layers: int = 0):
        try:
            from llama_cpp import Llama  # lazy import keeps the base install tiny
        except ImportError:
            raise ModelUnavailable("llama-cpp-python is not installed. Install it with: pip install llama-cpp-python")
        try:
            self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                             n_gpu_layers=n_gpu_layers, verbose=False)
        except Exception as e:
            raise ModelUnavailable(f"Could not load the model file '{model_path}' ({type(e).__name__}: {e}). "
                                   "Check the path and that the file is a valid .gguf model.")

    def generate(self, prompt: str) -> str:
        # chat API: llama.cpp applies the model's own chat template, which instruct models need
        out = self.llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                              max_tokens=256, temperature=0.2)
        return out["choices"][0]["message"]["content"].strip()


def make_cloze(sentence: str, index: BM25Index) -> str:
    """Blank out the rarest (most informative) word to make a guiding question."""
    words = re.findall(r"[\w\u1200-\u137F]+", sentence)
    cands = [w for w in words if len(w) > 3 and tokenize(w)]
    if not cands:
        return sentence
    key = max(cands, key=lambda w: index.idf.get(tokenize(w)[0], 0))
    return re.sub(rf"\b{re.escape(key)}\b", "_____", sentence, count=1)


class Tutor:
    def __init__(self, index: BM25Index, backend: Backend | None = None,
                 min_score: float = 1.0, rel: float = 0.5):
        # rel: keep only sources scoring at least rel x the best score (drops weak matches)
        self.index, self.backend, self.min_score, self.rel = index, backend, min_score, rel
        self.extractive = ExtractiveBackend()

    def ask(self, question: str, grade: int | None = None, lang: str = "English",
            subject: str | None = None) -> TutorReply:
        found = self.index.search(question, k=3, subject=subject, grade=grade)
        top_score = found[0].score if found else 0
        hits = [h for h in found if h.score >= self.min_score and h.score >= self.rel * top_score]
        if not hits:  # strict RAG: refuse instead of hallucinating
            return TutorReply(grounded=False,
                              answer="This is not in the textbook I have. Ask your teacher.")
        top = hits[0].passage
        cites = [h.passage.cite() for h in hits]
        sentence = self.extractive.best_sentence(question, top.text)
        hint = "Fill in the blank: " + make_cloze(sentence, self.index)
        if self.backend:
            ctx = "\n".join(f"[{c}] {h.passage.text}" for c, h in zip(cites, hits))
            try:
                raw = self.backend.generate(SOCRATIC_PROMPT.format(
                    level=f"Grade {grade}" if grade else "school", lang=lang, context=ctx, question=question))
            except Exception as e:  # the model must never break an answer: fall back to the textbook text
                log.warning("Language model failed (%s: %s); answering from the textbook instead.", type(e).__name__, e)
                return TutorReply(True, hint, top.text, cites, [h.passage for h in hits], ai_failed=True)
            claims = check(raw, [h.passage for h in hits])
            kept = " ".join(c.sentence for c in claims if c.supported)
            removed = sum(not c.supported for c in claims)
            answer = kept or top.text  # nothing verifiable: fall back to the textbook's own words
        else:
            answer, removed = top.text, 0
        return TutorReply(True, hint, answer, cites, [h.passage for h in hits], removed=removed)
