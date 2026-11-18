"""Teachers manage books from the browser: upload, list, replace, delete. No restart needed."""
import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from aitutor import ingest
from aitutor.api import create_app
from tests.test_import import make_pdf


def pdf_bytes(tmp_path):
    p = tmp_path / "src.pdf"; make_pdf(p)
    return p.read_bytes()


def blank_pdf() -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf); c.showPage(); c.save()
    return buf.getvalue()


@pytest.fixture
def web(tmp_path):
    c = TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(tmp_path / "data" / "p.jsonl"),
                              books_dir=str(tmp_path / "books"), admin_password="pw"))
    c.post("/api/login", json={"password": "pw"})
    return c, tmp_path


URL = "/admin/books/biology_g8_en.pdf?subject=biology&grade=8&lang=en"


def test_requires_admin_login(tmp_path):
    c = TestClient(create_app(str(tmp_path / "t.db"), books_dir=str(tmp_path / "b"), admin_password="pw"))
    assert c.get("/admin/books").status_code == 401
    assert c.put(URL, content=pdf_bytes(tmp_path)).status_code == 401
    assert c.delete("/admin/books/x.pdf").status_code == 401
    assert c.post("/admin/reindex").status_code == 401
    assert c.get("/teacher/library", follow_redirects=False).status_code == 303


def test_upload_is_usable_immediately_and_persists(web):
    c, tmp = web
    assert c.get("/api/library").json()["subjects"] == []
    r = c.put(URL, content=pdf_bytes(tmp))
    assert r.status_code == 200 and r.json()["passages"] > 0
    assert c.get("/api/library").json()["subjects"] == ["biology"]
    ans = c.post("/api/ask", json={"question": "How do plants make food?", "subject": "biology", "grade": 8}).json()
    assert ans["grounded"] and ans["citations"][0]["link"] == "/files/biology_g8_en.pdf#page=1"
    assert c.get("/files/biology_g8_en.pdf").content[:4] == b"%PDF"
    assert c.get(ans["citations"][0]["preview"]).status_code == 200
    row = c.get("/admin/books").json()["books"][0]
    assert (row["file"], row["subject"], row["grade"], row["lang"]) == ("biology_g8_en.pdf", "biology", 8, "en")
    assert row["passages"] == r.json()["passages"]
    # saved for the next start: manifest + imported passages
    assert json.loads((tmp / "books" / "manifest.json").read_text())["biology_g8_en.pdf"]["grade"] == 8
    assert len(ingest.load(tmp / "data" / "p.jsonl")) == r.json()["passages"]


def test_subject_is_cleaned_and_title_kept(web):
    c, tmp = web
    c.put("/admin/books/b.pdf?subject=Social%20Studies!&grade=7&title=My%20Book", content=pdf_bytes(tmp))
    row = c.get("/admin/books").json()["books"][0]
    assert row["subject"] == "socialstudies" and row["title"] == "My Book"


def test_replacing_a_book_does_not_duplicate_passages(web):
    c, tmp = web
    n1 = c.put(URL, content=pdf_bytes(tmp)).json()["passages"]
    n2 = c.put(URL, content=pdf_bytes(tmp)).json()["passages"]
    assert n1 == n2 and c.get("/admin/books").json()["books"][0]["passages"] == n1


@pytest.mark.parametrize("data,msg", [(b"this is not a pdf", "Could not read"), (None, "No readable text")])
def test_bad_files_are_rejected_and_library_unchanged(web, data, msg):
    c, tmp = web
    c.put(URL, content=pdf_bytes(tmp))
    before = c.get("/admin/books").json()
    r = c.put("/admin/books/bad_g8_en.pdf?subject=x", content=blank_pdf() if data is None else data)
    assert r.status_code == 422 and msg in r.json()["detail"]
    assert c.get("/admin/books").json() == before                       # nothing stored, good book untouched
    assert not (tmp / "books" / "bad_g8_en.pdf").exists()


@pytest.mark.parametrize("name", ["evil.txt", "..%2Fx.pdf", "a%2Fb.pdf", ".hidden.pdf", "x.exe"])
def test_unsafe_names_rejected(web, name):
    c, tmp = web
    assert c.put(f"/admin/books/{name}?subject=x", content=pdf_bytes(tmp)).status_code in (400, 404, 405)
    assert not list((tmp / "books").glob("*.exe")) and not (tmp / "x.pdf").exists()


def test_bad_language_and_grade_rejected(web):
    c, tmp = web
    assert c.put("/admin/books/a.pdf?subject=x&lang=xx", content=b"x").status_code == 400
    assert c.put("/admin/books/a.pdf?subject=x&grade=99", content=b"x").status_code == 422


def test_size_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("AITUTOR_MAX_UPLOAD_MB", "1")
    c = TestClient(create_app(str(tmp_path / "t.db"), books_dir=str(tmp_path / "b"),
                              passages_path=str(tmp_path / "p.jsonl"), admin_password="pw"))
    c.post("/api/login", json={"password": "pw"})
    r = c.put("/admin/books/big.pdf?subject=x", content=b"0" * 1_200_000)
    assert r.status_code == 413 and not (tmp_path / "b" / "big.pdf").exists()


def test_delete_removes_book_and_answers(web):
    c, tmp = web
    c.put(URL, content=pdf_bytes(tmp))
    assert c.delete("/admin/books/biology_g8_en.pdf").json() == {"deleted": "biology_g8_en.pdf"}
    assert not (tmp / "books" / "biology_g8_en.pdf").exists()
    assert "biology_g8_en.pdf" not in json.loads((tmp / "books" / "manifest.json").read_text())
    assert c.get("/admin/books").json()["books"] == []
    assert c.post("/api/ask", json={"question": "How do plants make food?"}).json()["grounded"] is False
    assert c.get("/files/biology_g8_en.pdf").status_code == 404
    assert c.delete("/admin/books/biology_g8_en.pdf").status_code == 404
    assert c.delete("/admin/books/..%2Fsecret.pdf").status_code in (400, 404)


def test_first_start_imports_books_already_in_folder(tmp_path):
    books = tmp_path / "books"; books.mkdir()
    (books / "biology_g8_en.pdf").write_bytes(pdf_bytes(tmp_path))
    c = TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(tmp_path / "p.jsonl"),
                              books_dir=str(books), admin_password="pw"))
    assert c.get("/api/library").json()["subjects"] == ["biology"]
    assert (tmp_path / "p.jsonl").exists()


def test_restart_keeps_uploaded_books(web):
    c, tmp = web
    c.put(URL, content=pdf_bytes(tmp))
    again = TestClient(create_app(str(tmp / "t2.db"), passages_path=str(tmp / "data" / "p.jsonl"),
                                  books_dir=str(tmp / "books"), admin_password="pw"))
    assert again.get("/api/library").json()["subjects"] == ["biology"]
    assert again.post("/api/ask", json={"question": "How do plants make food?"}).json()["grounded"]


def test_library_page_served_after_login(web):
    c, _ = web
    assert "Upload and import" in c.get("/teacher/library").text


def test_serve_command_accepts_every_documented_option(capsys):
    """Guards against a README/notebook flag the command line does not actually have."""
    from aitutor.cli import main
    with pytest.raises(SystemExit):
        main(["serve", "--help"])
    out = capsys.readouterr().out
    for flag in ("--admin-password", "--sync-key", "--student-code", "--pack", "--model", "--gpu-layers", "--port"):
        assert flag in out
