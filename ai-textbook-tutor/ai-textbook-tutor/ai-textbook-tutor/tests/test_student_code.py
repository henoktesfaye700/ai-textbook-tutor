"""An access code keeps a public site's books (and AI time) away from people who do not have the code."""
import pytest
from fastapi.testclient import TestClient
from aitutor.api import create_app
from tests.test_import import make_pdf


@pytest.fixture
def site(tmp_path):
    books = tmp_path / "books"; books.mkdir()
    make_pdf(books / "biology_g8_en.pdf")
    c = TestClient(create_app(str(tmp_path / "t.db"), passages_path=str(tmp_path / "p.jsonl"), books_dir=str(books),
                              admin_password="pw", student_code="ABC123", sync_key="k"))
    return c


def test_everything_students_see_is_locked(site):
    assert site.get("/api/library").status_code == 401
    assert site.post("/api/ask", json={"question": "plants"}).status_code == 401
    assert site.get("/files/biology_g8_en.pdf").status_code == 401
    assert site.get("/preview/anything.png").status_code == 401
    r = site.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/enter"
    assert site.get("/enter").status_code == 200 and "Access code" in site.get("/enter").text
    assert site.get("/health").status_code == 200            # monitoring still works


def test_correct_code_unlocks_and_wrong_is_rejected(site):
    assert site.post("/api/enter", json={"code": "nope"}).status_code == 401
    assert site.post("/api/enter", json={}).status_code == 401
    assert site.post("/api/enter", json={"code": "ABC123"}).status_code == 200
    assert site.get("/", follow_redirects=False).status_code == 200
    r = site.post("/api/ask", json={"question": "How do plants make food?"}).json()
    assert r["grounded"]
    assert site.get(r["citations"][0]["link"].split("#")[0]).status_code == 200
    assert site.get(r["citations"][0]["preview"]).status_code == 200


def test_header_and_admin_login_also_work(site, tmp_path):
    assert site.post("/api/ask", json={"question": "plants"}, headers={"X-Student-Code": "ABC123"}).status_code == 200
    assert site.post("/api/ask", json={"question": "plants"}, headers={"X-Student-Code": "x"}).status_code == 401
    site.post("/api/login", json={"password": "pw"})          # a logged-in teacher needs no code
    assert site.get("/api/library").status_code == 200


def test_student_cookie_never_grants_admin(site):
    site.post("/api/enter", json={"code": "ABC123"})
    assert site.get("/analytics/summary").status_code == 401
    assert site.get("/admin/books").status_code == 401
    assert site.get("/teacher", follow_redirects=False).status_code == 303
    student = site.cookies.get("aitutor_student")
    site.cookies.clear(); site.cookies.set("aitutor_session", student)   # try to pass it off as an admin cookie
    assert site.get("/analytics/summary").status_code == 401


def test_code_guessing_is_rate_limited(site):
    codes = [site.post("/api/enter", json={"code": f"g{i}"}).status_code for i in range(6)]
    assert codes == [401] * 5 + [429]
    assert site.post("/api/enter", json={"code": "ABC123"}).status_code == 429
    assert site.post("/api/login", json={"password": "pw"}).status_code == 200   # admins are not locked out by students


def test_devices_still_sync_with_the_sync_key_not_the_code(site):
    body = {"device_id": "d", "class_id": "8A", "events": []}
    assert site.post("/sync", json=body, headers={"X-Sync-Key": "k"}).status_code == 200
    assert site.post("/sync", json=body, headers={"X-Student-Code": "ABC123"}).status_code == 401


def test_no_code_means_open(tmp_path):
    c = TestClient(create_app(str(tmp_path / "t.db"), books_dir=str(tmp_path / "b"), passages_path=str(tmp_path / "p.jsonl"),
                              admin_password="pw"))
    assert c.get("/api/library").status_code == 200 and c.get("/", follow_redirects=False).status_code == 200
    assert c.post("/api/enter", json={"code": "whatever"}).json() == {"ok": True}
