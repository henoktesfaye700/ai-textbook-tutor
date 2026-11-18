"""Check an AI-written answer against the textbook, sentence by sentence.
A sentence is kept only if most of its content words appear in the retrieved passages."""
from __future__ import annotations
from dataclasses import dataclass
from .cleaning import split_sentences
from .models import Passage
from .retriever import tokenize


@dataclass
class Claim:
    sentence: str
    score: float
    supported: bool


def check(answer: str, passages: list[Passage], threshold: float = 0.7) -> list[Claim]:
    support = {w for p in passages for w in tokenize(p.text)}
    claims = []
    for sent in split_sentences(answer):
        words = set(tokenize(sent))
        score = len(words & support) / len(words) if words else 1.0
        claims.append(Claim(sent, round(score, 2), score >= threshold))
    return claims
