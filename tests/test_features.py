import json, unicodedata, zipfile
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
import io
from aitutor import pack as packlib
from aitutor.api import create_app
from aitutor.faithfulness import check
from aitutor.library import import_folder
from aitutor.models import Passage
from aitutor.retriever import BM25Index, normalize_geez, tokenize
from aitutor.tutor import Tutor
from aitutor import ingest
from tests.test_import import make_epub, make_pdf

NOW = 1_800_000_000


def app(tmp_path, **kw):
    return TestClient(create_app(str(tmp_path / "t.db"), clock=lambda: NOW, **kw))


# ------------------------------------------------------------ access control
def test_dashboard_requires_login(tmp_path):
    c = app(tmp_path, admin_password="pw")
    for path in ("/analytics/summary", "/analytics/report.pdf", "/dashboard/8A", "/analytics/export.csv"):
        assert c.get(path).status_code == 401
    assert c.get("/teacher", follow_redirects=False).headers["location"] == "/login"
    assert c.post("/api/login", json={"password": "nope"}).status_code == 401
    assert c.post("/api/login", json={"password": "pw"}).status_code == 200
    assert c.get("/analytics/summary").status_code == 200
    c.post("/api/logout")
    assert c.get("/analytics/summary").status_code == 401


def test_dashboard_disabled_without_password(tmp_path, monkeypatch):
    monkeypatch.delenv("AITUTOR_ADMIN_PASSWORD", raising=False)
    c = app(tmp_path)
    assert c.get("/analytics/summary").status_code == 403
    assert c.post("/api/login", json={"password": "anything"}).status_code == 403


def test_bearer_token_and_unicode_password(tmp_path):
    c = app(tmp_path, admin_password="pw")
    assert c.get("/analytics/summary", headers={"Authorization": "Bearer pw"}).status_code == 200
    assert c.get("/analytics/summary", headers={"Authorization": "Bearer bad"}).status_code == 401
    u = app(tmp_path, admin_password="ሰላም-pass")
    assert u.post("/api/login", json={"password": "ሰላም-pass"}).status_code == 200


def test_login_lockout_after_five_failures(tmp_path):
    c = app(tmp_path, admin_password="pw")
    codes = [c.post("/api/login", json={"password": "x"}).status_code for _ in range(6)]
    assert codes == [401] * 5 + [429]
    assert c.post("/api/login", json={"password": "pw"}).status_code == 429   # even the right one waits


def test_tampered_or_expired_session_rejected(tmp_path):
    t = [NOW]
    c = TestClient(create_app(str(tmp_path / "t.db"), clock=lambda: t[0], admin_password="pw", secret="s"))
    c.post("/api/login", json={"password": "pw"})
    good = c.cookies.get("aitutor_session")
    assert c.get("/analytics/summary").status_code == 200
    exp, sig = good.split(".")
    c.cookies.set("aitutor_session", f"{int(exp) + 99999}.{sig}")        # forged expiry
    assert c.get("/analytics/summary").status_code == 401
    c.cookies.set("aitutor_session", good)
    t[0] = NOW + 13 * 3600                                           # past the 12 h lifetime
    assert c.get("/analytics/summary").status_code == 401


def test_sync_key_protects_uploads(tmp_path):
    c = app(tmp_path, admin_password="pw", sync_key="k1")
    body = {"device_id": "d", "class_id": "8A", "events": []}
    assert c.post("/sync", json=body).status_code == 401
    assert c.post("/sync", json=body, headers={"X-Sync-Key": "wrong"}).status_code == 401
    assert c.post("/sync", json=body, headers={"X-Sync-Key": "k1"}).status_code == 200


# ------------------------------------------------------------ PDF report
def test_pdf_report_contains_numbers(tmp_path):
    c = app(tmp_path, admin_password="pw")
    ev = [{"topic": "Cell", "subject": "biology", "question_id": f"q{i}", "correct": i < 3, "ts": NOW - i}
          for i in range(10)]
    c.post("/sync", json={"device_id": "d1", "class_id": "8A", "school_id": "Hawassa", "project_id": "P", "events": ev})
    c.post("/api/login", json={"password": "pw"})
    r = c.get("/analytics/report.pdf", params={"school": "Hawassa"})
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(r.content)).pages)
    assert "AI Tutor report: Hawassa" in text and "8A" in text and "biology" in text and "30%" in text


# ------------------------------------------------------------ content packs
@pytest.fixture
def lib(tmp_path):
    books = tmp_path / "books"; books.mkdir()
    make_pdf(books / "biology_g8_en.pdf"); make_epub(books / "physics.epub")
    ps, _ = import_folder(books)
    ingest.save(ps, tmp_path / "passages.jsonl")
    return tmp_path, books


def test_pack_roundtrip_and_version_rules(lib, tmp_path_factory):
    src, books = lib
    out = src / "school.tutorpack"
    m = packlib.build(src / "passages.jsonl", books, out, version=2, key="K")
    assert m["version"] == 2 and "books/biology_g8_en.pdf" in m["files"] and m["signature"]
    dst = tmp_path_factory.mktemp("device")
    packlib.install(out, dst / "data", dst / "books", key="K")
    assert (dst / "books" / "biology_g8_en.pdf").read_bytes()[:4] == b"%PDF"
    assert ingest.load(dst / "data" / "passages.jsonl") == ingest.load(src / "passages.jsonl")
    with pytest.raises(packlib.PackError, match="not newer"):
        packlib.install(out, dst / "data", dst / "books", key="K")
    packlib.install(out, dst / "data", dst / "books", key="K", force=True)


def test_pack_rejects_tampering_wrong_key_and_unsafe_paths(lib, tmp_path):
    src, books = lib
    out = src / "p.tutorpack"
    packlib.build(src / "passages.jsonl", books, out, version=1, key="K")
    with pytest.raises(packlib.PackError, match="signature"):
        packlib.install(out, tmp_path / "d", tmp_path / "b", key="WRONG")
    bad = src / "bad.tutorpack"                                        # change one file, keep the manifest
    with zipfile.ZipFile(out) as zin, zipfile.ZipFile(bad, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, b"tampered" if n == "passages.jsonl" else zin.read(n))
    with pytest.raises(packlib.PackError, match="checksum"):
        packlib.install(bad, tmp_path / "d", tmp_path / "b", key="K")
    assert not (tmp_path / "d" / "passages.jsonl").exists()      # nothing written when verification fails
    evil = src / "evil.tutorpack"
    files = {"../evil.txt": packlib._sha(b"x")}
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": 1, "version": 1, "created": "", "files": files}))
        z.writestr("../evil.txt", b"x")
    with pytest.raises(packlib.PackError, match="unsafe"):
        packlib.install(evil, tmp_path / "d", tmp_path / "b")
    unsigned = src / "u.tutorpack"
    packlib.build(src / "passages.jsonl", books, unsigned, version=1)
    with pytest.raises(packlib.PackError, match="signature"):     # key given but pack unsigned
        packlib.install(unsigned, tmp_path / "d2", tmp_path / "b2", key="K")


def test_server_publishes_pack_with_key(lib, tmp_path):
    src, books = lib
    packlib.build(src / "passages.jsonl", books, src / "p.tutorpack", version=7)
    c = TestClient(create_app(str(tmp_path / "t.db"), admin_password="pw", sync_key="k",
                              pack_path=str(src / "p.tutorpack")))
    assert c.get("/content/version").json() == {"version": 7}
    assert c.get("/content/pack").status_code == 401
    r = c.get("/content/pack", headers={"X-Sync-Key": "k"})
    assert r.status_code == 200 and zipfile.is_zipfile(io.BytesIO(r.content))


# ------------------------------------------------------------ Ge'ez spelling variants
@pytest.mark.parametrize("variant,canonical", [(0x1210, 0x1200), (0x1280, 0x1200), (0x1220, 0x1230),
                                                (0x12D0, 0x12A0), (0x1340, 0x1338)])
def test_geez_variant_series_map_to_canonical(variant, canonical):
    for k in range(7):                       # all seven vowel orders
        assert normalize_geez(chr(variant + k)) == chr(canonical + k)
    assert unicodedata.name(chr(variant)).startswith("ETHIOPIC SYLLABLE")


def test_search_matches_either_amharic_spelling():
    sese, sa = "\u1220\u120b\u121d", "\u1230\u120b\u121d"            # ሠላም / ሰላም (peace)
    assert tokenize(sese) == tokenize(sa)
    p = Passage("a", "t", "u", 1, f"{sese} \u12e8\u1235\u120b\u121d")
    assert BM25Index([p]).search(sa)[0].passage.id == "a"


# ------------------------------------------------------------ faithfulness check
class FakeLLM:
    def __init__(self, text): self.text = text
    def generate(self, prompt): return self.text


def passage_index():
    ps = [Passage("d", "Density", "Unit 5", 70, "Density is mass divided by volume. An object floats if its density is lower than water."),
          Passage("c", "Cell", "Unit 2", 28, "The nucleus holds the genetic material and controls the cell."),
          Passage("w", "Water", "Unit 3", 41, "Rising vapour cools and condenses into clouds before falling as rain.")]
    return BM25Index(ps), ps[:1]


def test_faithfulness_scores_sentences():
    _, ps = passage_index()
    claims = check("Density is mass divided by volume. The moon is made of green cheese.", ps)
    assert [c.supported for c in claims] == [True, False]


def test_tutor_drops_unsupported_ai_sentences():
    idx, _ = passage_index()
    llm = FakeLLM("Density is mass divided by volume. The moon is made of green cheese.")
    r = Tutor(idx, backend=llm).ask("What is density?")
    assert "cheese" not in r.answer and "mass divided by volume" in r.answer and r.removed == 1


def test_tutor_falls_back_to_textbook_when_nothing_verifiable():
    idx, ps = passage_index()
    r = Tutor(idx, backend=FakeLLM("Volcanoes erupt molten rainbow glitter.")).ask("What is density?")
    assert r.answer == ps[0].text and r.removed == 1


def test_cli_reports_rejected_pack_clearly(lib, tmp_path):
    from aitutor.cli import main
    src, books = lib
    packlib.build(src / "passages.jsonl", books, src / "p.tutorpack", version=1, key="K")
    with pytest.raises(SystemExit) as e:
        main(["pack", "install", str(src / "p.tutorpack"), "--data", str(tmp_path / "d"),
              "--books", str(tmp_path / "b"), "--sign-key", "WRONG"])
    assert "Pack rejected" in str(e.value) and "signature" in str(e.value)
