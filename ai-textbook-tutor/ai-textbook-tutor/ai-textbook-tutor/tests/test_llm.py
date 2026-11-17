"""The real model cannot run in CI, so llama_cpp is replaced by a stand-in that records how AI Textbook Tutor calls it."""
import sys
import types
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from aitutor import ingest
from aitutor.api import create_app
from aitutor.cli import main
from aitutor.models import Passage
from aitutor.retriever import BM25Index
from aitutor.tutor import LlamaCppBackend, Tutor

FIXTURE = Path(__file__).parent / "fixtures" / "raw_textbook.txt"
GOOD = "Density is mass divided by volume."
BAD = "Volcanoes are powered by tiny dragons."


@pytest.fixture
def fake_llama(monkeypatch):
    calls = {"init": None, "chat": []}

    class Llama:
        def __init__(self, **kw):
            calls["init"] = kw

        def create_chat_completion(self, **kw):
            calls["chat"].append(kw)
            return {"choices": [{"message": {"content": f"  {GOOD} {BAD}  "}}]}

    mod = types.ModuleType("llama_cpp")
    mod.Llama = Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return calls


@pytest.fixture
def passages_file(tmp_path):
    f = tmp_path / "passages.jsonl"
    ingest.save(ingest.parse_raw(FIXTURE.read_text(encoding="utf-8")), f)
    return f


def test_backend_uses_chat_api_and_options(fake_llama):
    b = LlamaCppBackend("model.gguf", n_gpu_layers=7)
    out = b.generate("hello")
    assert out == f"{GOOD} {BAD}"                                   # whitespace trimmed
    assert fake_llama["init"]["model_path"] == "model.gguf" and fake_llama["init"]["n_gpu_layers"] == 7
    assert fake_llama["init"]["verbose"] is False
    call = fake_llama["chat"][0]
    assert call["messages"] == [{"role": "user", "content": "hello"}] and call["max_tokens"] == 256


def test_prompt_has_language_level_and_cited_excerpts():
    seen = []

    class Spy:
        def generate(self, prompt):
            seen.append(prompt); return GOOD
    ps = [Passage("d", "Density", "Unit 5", 70, "Density is mass divided by volume."),
          Passage("c", "Cell", "Unit 2", 28, "The nucleus holds the genetic material."),
          Passage("w", "Water", "Unit 3", 41, "Vapour cools and condenses into clouds.")]
    Tutor(BM25Index(ps), Spy()).ask("What is density?", grade=8, lang="Amharic")
    p = seen[0]
    assert "Grade 8" in p and "Amharic" in p and "[Unit 5, Density, p. 70]" in p and "What is density?" in p


def test_server_with_model_removes_unsupported_sentences(fake_llama, passages_file, tmp_path):
    c = TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(passages_file),
                              model_path="m.gguf", gpu_layers=3, admin_password="pw"))
    assert fake_llama["init"]["n_gpu_layers"] == 3
    assert c.get("/api/library").json()["ai"] is True
    r = c.post("/api/ask", json={"question": "What is density?", "lang": "Somali"}).json()
    assert r["ai"] and r["removed"] == 1 and GOOD in r["answer"] and "dragons" not in r["answer"]
    assert "Somali" in fake_llama["chat"][0]["messages"][0]["content"]   # language reached the model


def test_server_without_model_is_extractive(passages_file, tmp_path):
    c = TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(passages_file), admin_password="pw"))
    assert c.get("/api/library").json()["ai"] is False
    r = c.post("/api/ask", json={"question": "What is density?"}).json()
    assert r["grounded"] and r["ai"] is False and r["removed"] == 0


def test_server_does_not_call_model_for_out_of_scope_questions(fake_llama, passages_file, tmp_path):
    c = TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(passages_file), model_path="m.gguf"))
    r = c.post("/api/ask", json={"question": "Who won the football match yesterday?"}).json()
    assert r["grounded"] is False and fake_llama["chat"] == []     # refuses before spending model time


def test_cli_ask_with_model(fake_llama, passages_file, capsys):
    main(["ask", "What is density?", "--passages", str(passages_file), "--model", "m.gguf",
          "--lang", "Amharic", "--gpu-layers", "2"])
    out = capsys.readouterr().out
    assert GOOD in out and "dragons" not in out and "1 unsupported sentence(s) removed" in out
    assert fake_llama["init"]["n_gpu_layers"] == 2
