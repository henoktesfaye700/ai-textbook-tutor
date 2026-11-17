"""Measure retrieval quality on YOUR books: recall@1 and recall@3 on questions you label.
  python scripts/evaluate.py --passages data/passages.jsonl --questions data/eval_questions.jsonl
Each line of the questions file: {"q": "...", "topic": "<chapter title the answer is in>"}
Exits non-zero if recall@3 is below --min (default 0.9)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aitutor import ingest
from aitutor.retriever import BM25Index

ap = argparse.ArgumentParser()
ap.add_argument("--passages", default="data/passages.jsonl")
ap.add_argument("--questions", default="data/eval_questions.jsonl")
ap.add_argument("--min", type=float, default=0.9)
a = ap.parse_args()
index = BM25Index(ingest.load(Path(a.passages)))
rows = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
r1 = r3 = 0
for r in rows:
    topics = [h.passage.topic for h in index.search(r["q"], k=3)]
    r1 += bool(topics) and topics[0] == r["topic"]
    r3 += r["topic"] in topics
    if not topics or topics[0] != r["topic"]:
        print("MISS@1:", r["q"], "->", topics)
n = len(rows)
print(f"recall@1 = {r1/n:.2f}   recall@3 = {r3/n:.2f}   (n={n})")
sys.exit(0 if r3 / n >= a.min else 1)
