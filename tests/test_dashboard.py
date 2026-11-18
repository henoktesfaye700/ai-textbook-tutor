import csv, io, re, shutil, sqlite3, subprocess
import pytest
from fastapi.testclient import TestClient
from aitutor import ingest
from aitutor.api import create_app
from aitutor.library import import_folder
from tests.test_import import make_epub, make_pdf

NOW = 1_800_000_000
DAY = 86400


def ev(topic, subject, qid, ok, days_ago, n=0):
    return {"topic": topic, "subject": subject, "question_id": qid, "correct": ok, "ts": NOW - days_ago * DAY - n}


@pytest.fixture
def client(tmp_path):
    c = TestClient(create_app(str(tmp_path / "t.db"), clock=lambda: NOW, admin_password="pw"))
    def post(dev, cls, school, project, events):
        assert c.post("/sync", json={"device_id": dev, "class_id": cls, "school_id": school,
                                     "project_id": project, "events": events}).status_code == 200
    # school A / project P1: two students; school B / project P2: one student (inactive, weak)
    post("a1" * 8, "8A", "School A", "P1", [ev("Cell", "biology", f"q{i}", i % 2 == 0, 1, i) for i in range(6)])
    post("a2" * 8, "8A", "School A", "P1", [ev("Density", "physics", f"p{i}", True, 2, i) for i in range(6)])
    post("b1" * 8, "7B", "School B", "P2", [ev("Density", "physics", f"p{i}", False, 30, i) for i in range(6)])
    assert c.post("/api/login", json={"password": "pw"}).status_code == 200
    return c


def test_summary_counts_students_and_filters(client):
    s = client.get("/analytics/summary").json()
    assert (s["students"], s["attempts"], s["schools"], s["classes"], s["subjects"]) == (3, 18, 2, 2, 2)
    assert s["active_7d"] == 2                       # b1 last active 30 days ago
    only = client.get("/analytics/summary", params={"school": "School A"}).json()
    assert only["students"] == 2 and only["attempts"] == 12
    assert client.get("/analytics/summary", params={"project": "P2", "subject": "physics"}).json()["accuracy"] == 0
    assert client.get("/analytics/summary", params={"days": 7}).json()["students"] == 2


@pytest.mark.parametrize("dim,expected", [("school", {"School A", "School B"}), ("project", {"P1", "P2"}),
                                          ("class", {"8A", "7B"}), ("subject", {"biology", "physics"}),
                                          ("topic", {"Cell", "Density"})])
def test_by_dimension(client, dim, expected):
    rows = client.get(f"/analytics/by/{dim}").json()["rows"]
    assert {r["name"] for r in rows} == expected
    assert sum(r["attempts"] for r in rows) == 18


def test_by_dimension_rejects_unknown_column(client):
    assert client.get("/analytics/by/device_id;DROP TABLE events").status_code in (404, 422)
    assert client.get("/analytics/by/password").status_code == 422


def test_needs_review_and_students_at_risk(client):
    school = {r["name"]: r for r in client.get("/analytics/by/school").json()["rows"]}
    assert school["School B"]["needs_review"] and not school["School A"]["needs_review"]
    st = client.get("/analytics/students").json()
    assert st["total"] == 3 and st["at_risk_count"] == 1
    flagged = st["students"][0]
    assert set(flagged["at_risk"]) == {"low accuracy", "inactive 14+ days"} and flagged["student"] == "b1b1b1b1"


def test_timeseries_and_hardest_questions(client):
    days = client.get("/analytics/timeseries").json()["days"]
    assert sum(d["attempts"] for d in days) == 18 and len(days) == 3
    q = client.get("/analytics/questions", params={"min_attempts": 2}).json()["questions"]
    assert q[0]["accuracy"] <= q[-1]["accuracy"]


def test_filter_options_and_csv(client):
    f = client.get("/analytics/filters").json()
    assert f["school"] == ["School A", "School B"] and f["project"] == ["P1", "P2"]
    r = client.get("/analytics/export.csv", params={"dim": "school"})
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0][:3] == ["school", "students", "attempts"] and len(rows) == 3
    assert "attachment" in r.headers["content-disposition"]


def test_old_database_is_migrated(tmp_path):
    path = str(tmp_path / "old.db")
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE events(device_id TEXT, class_id TEXT, topic TEXT, question_id TEXT,
                   correct INTEGER, ts INTEGER, UNIQUE(device_id, question_id, ts))""")
    con.execute("INSERT INTO events VALUES ('d','8A','Cell','q',1,?)", (NOW,)); con.commit(); con.close()
    c = TestClient(create_app(path, clock=lambda: NOW, admin_password="pw"))
    c.post("/api/login", json={"password": "pw"})
    assert c.get("/analytics/summary").json()["attempts"] == 1
    assert c.get("/analytics/by/school").json()["rows"][0]["name"] == "(none)"


# ------------------------------------------------------------ tutor with page links and previews
@pytest.fixture
def lib_client(tmp_path):
    books = tmp_path / "books"; books.mkdir()
    make_pdf(books / "biology_g8_en.pdf"); make_epub(books / "physics.epub")
    (books / "manifest.json").write_text('{"physics.epub": {"subject": "physics", "grade": 7}}')
    ps, _ = import_folder(books)
    ingest.save(ps, tmp_path / "p.jsonl")
    (tmp_path / "secret.txt").write_text("secret")
    return TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(tmp_path / "p.jsonl"),
                                 books_dir=str(books), admin_password="pw"))


def test_ask_returns_pdf_link_to_physical_page(lib_client):
    r = lib_client.post("/api/ask", json={"question": "How do plants make food?", "subject": "biology"}).json()
    c = r["citations"][0]
    assert r["grounded"] and c["page"] == 11 and c["pdf_page"] == 1   # printed p.11 is PDF page 1
    assert c["link"] == "/files/biology_g8_en.pdf#page=1" and c["preview"].startswith("/preview/")


def test_epub_citation_has_no_pdf_link(lib_client):
    c = lib_client.post("/api/ask", json={"question": "Newton second law", "subject": "physics"}).json()["citations"][0]
    assert c["link"] is None and c["preview"] is None and c["excerpt"]


def test_file_endpoint_serves_pdf_and_blocks_traversal(lib_client):
    r = lib_client.get("/files/biology_g8_en.pdf")
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    for bad in ("../secret.txt", "..%2Fsecret.txt", "secret.txt", "nope.pdf"):
        assert lib_client.get(f"/files/{bad}").status_code == 404


def test_preview_png_with_highlight(lib_client):
    q = "How do plants make food?"
    c = lib_client.post("/api/ask", json={"question": q}).json()["citations"][0]
    plain = lib_client.get(c["preview"].split("?")[0])
    marked = lib_client.get(c["preview"])
    assert plain.status_code == marked.status_code == 200 and marked.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert plain.content != marked.content            # highlight boxes were actually drawn
    assert lib_client.get("/preview/nope.png").status_code == 404


def test_library_and_pages(lib_client):
    lib = lib_client.get("/api/library").json()
    assert set(lib["subjects"]) == {"biology", "physics"} and lib["grades"] == [7, 8]
    assert lib_client.get("/teacher", follow_redirects=False).status_code == 303   # not logged in
    lib_client.post("/api/login", json={"password": "pw"})
    assert "AI Tutor dashboard" in lib_client.get("/teacher").text and "AI Textbook Tutor" in lib_client.get("/").text


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
@pytest.mark.parametrize("page", ["index", "teacher", "login", "library", "enter"])
def test_page_scripts_have_valid_syntax(tmp_path, page):
    from aitutor.api import WEB
    js = re.search(r"<script>(.*)</script>", (WEB / f"{page}.html").read_text(encoding="utf-8"), re.S).group(1)
    f = tmp_path / "x.js"; f.write_text(js)
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
