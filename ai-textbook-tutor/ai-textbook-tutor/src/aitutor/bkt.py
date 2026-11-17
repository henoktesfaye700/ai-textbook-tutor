"""Bayesian Knowledge Tracing: estimates P(student has mastered a skill)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Skill:
    p_known: float = 0.2
    p_learn: float = 0.15
    p_slip: float = 0.10
    p_guess: float = 0.25

    def update(self, correct: bool) -> float:
        k, s, g = self.p_known, self.p_slip, self.p_guess
        if correct:
            post = k * (1 - s) / (k * (1 - s) + (1 - k) * g)
        else:
            post = k * s / (k * s + (1 - k) * (1 - g))
        self.p_known = post + (1 - post) * self.p_learn
        return self.p_known


@dataclass
class Student:
    skills: dict[str, Skill] = field(default_factory=dict)

    def observe(self, topic: str, correct: bool) -> float:
        return self.skills.setdefault(topic, Skill()).update(correct)

    def next_topic(self, topics: list[str], mastery: float = 0.95) -> str | None:
        """Practise the weakest topic that is not yet mastered."""
        open_ = [(self.skills.get(t, Skill()).p_known, t) for t in topics]
        open_ = [x for x in open_ if x[0] < mastery]
        return min(open_)[1] if open_ else None

    def difficulty(self, topic: str) -> int:
        return 2 if self.skills.get(topic, Skill()).p_known > 0.6 else 1
