"""Raw OCR text -> cleaned passages (JSONL). Input format:
#UNIT <unit> | <topic> | <page> | <lang>
<raw text...>
"""
from __future__ import annotations
import json
from pathlib import Path
from .cleaning import chunk
from .models import Passage


def parse_raw(raw: str) -> list[Passage]:
    out: list[Passage] = []
    for block in raw.split("#UNIT")[1:]:
        header, _, body = block.partition("\n")
        parts = [p.strip() for p in header.split("|")]
        unit, topic, page = parts[0], parts[1], int(parts[2])
        lang = parts[3] if len(parts) > 3 else "en"
        out += chunk(body, unit, topic, page, lang)
    return out


def save(passages: list[Passage], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")


def load(path: Path) -> list[Passage]:
    with path.open(encoding="utf-8") as f:
        return [Passage(**json.loads(line)) for line in f if line.strip()]
