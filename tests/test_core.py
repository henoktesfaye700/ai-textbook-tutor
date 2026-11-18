import json
from pathlib import Path
from fastapi.testclient import TestClient
from aitutor import ingest
from aitutor.api import create_app
from aitutor.bkt import Student
from aitutor.cleaning import normalize, chunk
from aitutor.retriever import BM25Index
from aitutor.tutor import Tutor

RAW = (Path(__file__).parent / "fixtures" / "raw_textbook.txt").read_text(encoding="utf-8")


def test_normalize_fixes_hyphen_and_page_numbers():
    out = normalize("Inside chloro-\nplasts, light.\n12\nNext.")
    assert "chloroplasts" in out and " 12 " not in out


def test_normalize_geez_punctuation():
    assert "።" not in normalize("ሰላም። ደህና")


def test_chunk_keeps_citation_fields():
    ps = chunk("One. Two. Three.", "Unit 9", "Test", 5, max_words=2)
    assert len(ps) >= 2 and all(p.page == 5 and p.unit == "Unit 9" for p in ps)


def test_retrieval_finds_topic():
    idx = BM25Index(ingest.parse_raw(RAW))
    assert idx.search("what is the pH of pure water")[0].passage.topic == "Acids and bases"


def test_tutor_is_grounded_and_cites():
    t = Tutor(BM25Index(ingest.parse_raw(RAW)))
    r = t.ask("How do plants make food?")
    assert r.grounded and "_____" in r.hint and "p. 12" in r.citations[0]


def test_tutor_refuses_out_of_scope():
    t = Tutor(BM25Index(ingest.parse_raw(RAW)))
    assert not t.ask("Who won the football match yesterday?").grounded


def test_bkt_learns_and_picks_weakest():
    s = Student()
    for _ in range(4):
        s.observe("density", True)
    s.observe("circuits", False)
    assert s.skills["density"].p_known > 0.8
    assert s.next_topic(["density", "circuits"]) == "circuits"


def test_api_sync_is_idempotent_and_dashboard_flags(tmp_path):
    c = TestClient(create_app(str(tmp_path / "t.db"), admin_password="pw"))
    c.post("/api/login", json={"password": "pw"})
    ev = [{"topic": "Density", "question_id": f"q{i}", "correct": False, "ts": i} for i in range(6)]
    body = {"device_id": "d1", "class_id": "8A", "events": ev}
    assert c.post("/sync", json=body).json()["accepted"] == 6
    assert c.post("/sync", json=body).json() == {"accepted": 0, "duplicates": 6}
    d = c.get("/dashboard/8A").json()
    assert d["topics"][0]["needs_review"] is True
