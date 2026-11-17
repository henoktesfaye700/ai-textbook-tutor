"""Auto-generated fill-in-the-blank quizzes + an adaptive session driven by BKT."""
from __future__ import annotations
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from .bkt import Student
from .cleaning import split_sentences
from .models import Passage
from .retriever import BM25Index, tokenize

WORD = re.compile(r"[A-Za-z\u1200-\u137F]{5,}")


@dataclass
class Question:
    id: str
    topic: str
    subject: str
    prompt: str
    options: list[str]
    answer: int
    source: str


def generate(passages: list[Passage], index: BM25Index, seed: int = 0) -> list[Question]:
    """One multiple-choice question per passage. Distractors come from the same subject."""
    rng = random.Random(seed)
    vocab: dict[str, set[str]] = {}
    for p in passages:
        vocab.setdefault(p.subject, set()).update(w.lower() for w in WORD.findall(p.text))

    def weight(w: str) -> float:
        t = tokenize(w)
        return index.idf.get(t[0], 0.0) if t else 0.0

    out = []
    for p in passages:
        best = None
        for sent in split_sentences(p.text):
            words = WORD.findall(sent)
            if len(sent.split()) < 6 or not words:
                continue
            key = max(words, key=weight)
            if best is None or weight(key) > best[0]:
                best = (weight(key), sent, key)
        if not best:
            continue
        _, sent, key = best
        used = {w.lower() for w in WORD.findall(sent)}
        pool = sorted(w for w in vocab[p.subject] if w not in used and abs(len(w) - len(key)) <= 4)
        if len(pool) < 3:
            continue
        options = rng.sample(pool, 3) + [key.lower()]
        rng.shuffle(options)
        qid = hashlib.sha1(f"{p.id}|{sent}".encode()).hexdigest()[:10]
        out.append(Question(qid, p.topic, p.subject, re.sub(rf"\b{re.escape(key)}\b", "_____", sent, count=1),
                            options, options.index(key.lower()), p.cite()))
    return out


class QuizSession:
    """Always practise the weakest topic (lowest estimated mastery) first."""

    def __init__(self, questions: list[Question], student: Student | None = None):
        self.remaining = list(questions)
        self.student = student or Student()
        self.events: list[dict] = []

    def next(self) -> Question | None:
        topic = self.student.next_topic(sorted({q.topic for q in self.remaining}))
        return next((q for q in self.remaining if q.topic == topic), None)

    def answer(self, q: Question, choice: int) -> bool:
        ok = choice == q.answer
        self.student.observe(q.topic, ok)
        self.remaining.remove(q)
        self.events.append({"topic": q.topic, "question_id": q.id, "correct": ok, "ts": int(time.time()),
                            "subject": q.subject})
        return ok

    def mastery(self) -> dict[str, float]:
        return {t: round(s.p_known, 2) for t, s in self.student.skills.items()}


def save_outbox(path: Path, events: list[dict], device_id: str) -> None:
    """Events wait here until the device is online (offline-first)."""
    data = json.loads(path.read_text()) if path.exists() else {"device_id": device_id, "events": []}
    data["events"] += events
    path.write_text(json.dumps(data, indent=1))
