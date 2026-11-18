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
from aitutor.tutor import LlamaCppBackend, ModelUnavailable, Tutor

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


# ---- what happens when the language model is not available -------------------------------------------------
def no_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # makes "import llama_cpp" raise ImportError


def test_engine_not_installed_gives_a_clear_error(monkeypatch):
    no_engine(monkeypatch)
    with pytest.raises(ModelUnavailable, match="pip install llama-cpp-python"):
        LlamaCppBackend("model.gguf")


def test_bad_model_file_gives_a_clear_error(monkeypatch):
    class Llama:
        def __init__(self, **kw):
            raise ValueError("Model path does not exist: missing.gguf")
    mod = types.ModuleType("llama_cpp"); mod.Llama = Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    with pytest.raises(ModelUnavailable, match="missing.gguf.*valid .gguf"):
        LlamaCppBackend("missing.gguf")


def test_server_stays_up_without_the_engine(monkeypatch, passages_file, tmp_path, caplog):
    no_engine(monkeypatch)
    with caplog.at_level("WARNING", logger="aitutor"):
        c = TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(passages_file), model_path="m.gguf"))
    assert "pip install llama-cpp-python" in caplog.text and "WITHOUT the AI model" in caplog.text
    assert c.get("/api/library").json()["ai"] is False
    r = c.post("/api/ask", json={"question": "What is density?"}).json()
    assert r["grounded"] and r["ai"] is False and "mass divided by volume" in r["answer"]


def test_model_failing_mid_answer_falls_back_to_the_textbook(passages_file, tmp_path, caplog):
    class Broken:
        def generate(self, prompt):
            raise RuntimeError("out of memory")
    c = TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(passages_file), backend=Broken()))
    with caplog.at_level("WARNING", logger="aitutor"):
        res = c.post("/api/ask", json={"question": "What is density?"})
    assert res.status_code == 200                                    # a model error never becomes a server error
    r = res.json()
    assert r["grounded"] and r["ai"] is False and "mass divided by volume" in r["answer"] and r["citations"]
    assert "out of memory" in caplog.text


def test_cli_reports_a_missing_engine_cleanly(monkeypatch, passages_file):
    no_engine(monkeypatch)
    with pytest.raises(SystemExit) as e:
        main(["ask", "What is density?", "--passages", str(passages_file), "--model", "m.gguf"])
    assert "pip install llama-cpp-python" in str(e.value) and "without --model" in str(e.value)


def test_cli_without_a_model_never_needs_the_engine(monkeypatch, passages_file, capsys):
    no_engine(monkeypatch)
    main(["ask", "What is density?", "--passages", str(passages_file)])
    out = capsys.readouterr().out
    assert "ANSWER" in out and "mass divided by volume" in out and "AI answer" not in out
