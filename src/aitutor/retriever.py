"""Dependency-free BM25 retriever. Runs on a low-end phone: no GPU, no network."""
from __future__ import annotations
import math
import re
from collections import Counter
from .models import Hit, Passage

STOP = set("a an the is are of to in and or for on with it its this that what how why does do which".split())
_TOKEN = re.compile(r"[\w\u1200-\u137F]+", re.UNICODE)  # \u1200-\u137F = Ge'ez block


# Amharic has letters that sound identical and are used interchangeably in spelling.
# Map each variant series (7 vowel orders) onto one canonical series so searches match either spelling.
_GEEZ_VARIANTS = {0x1210: 0x1200, 0x1280: 0x1200,   # HHA, XA   -> HA
                  0x1220: 0x1230,                    # SZA       -> SA
                  0x12D0: 0x12A0,                    # PHARYNGEAL -> GLOTTAL
                  0x1340: 0x1338}                    # TZA       -> TSA
_GEEZ_TABLE = {src + k: dst + k for src, dst in _GEEZ_VARIANTS.items() for k in range(7)}


def normalize_geez(text: str) -> str:
    return text.translate(_GEEZ_TABLE)


def stem(w: str) -> str:
    for suf in ("ing", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _TOKEN.findall(normalize_geez(text).lower()) if w not in STOP]


class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages, self.k1, self.b = passages, k1, b
        self.tf = [Counter(tokenize(p.text + " " + p.topic)) for p in passages]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter(w for c in self.tf for w in c)
        n = len(passages)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int = 3, subject: str | None = None,
               grade: int | None = None) -> list[Hit]:
        """Optionally restrict to one subject and grade (grade 0 = unspecified, always kept)."""
        q = tokenize(query)
        hits = []
        for i, p in enumerate(self.passages):
            if subject and p.subject.lower() != subject.lower():
                continue
            if grade and p.grade not in (0, grade):
                continue
            s = 0.0
            for w in q:
                f = self.tf[i].get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
            if s > 0:
                hits.append(Hit(p, s))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
