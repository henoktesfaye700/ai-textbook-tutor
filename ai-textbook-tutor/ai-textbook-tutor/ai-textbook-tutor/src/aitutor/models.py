from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Passage:
    id: str
    topic: str
    unit: str
    page: int
    text: str
    lang: str = "en"
    subject: str = "general"
    grade: int = 0  # 0 = unspecified
    book: str = ""
    source: str = ""  # file name of the book (for page links)
    phys: int = 0     # physical PDF page (1-based); differs from the printed page number

    def cite(self) -> str:
        parts = [self.unit] + ([self.topic] if self.topic and self.topic != self.unit else [])
        s = ", ".join(parts)
        return f"{s}, p. {self.page}" if self.page else s


@dataclass
class Hit:
    passage: Passage
    score: float


@dataclass
class TutorReply:
    grounded: bool
    hint: str = ""
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    removed: int = 0  # sentences from an AI answer dropped because the textbook did not support them
