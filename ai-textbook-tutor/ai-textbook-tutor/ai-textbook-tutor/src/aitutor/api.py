"""AI Textbook Tutor server: sync API, analytics for dashboards, and the student/teacher web pages.
Students work offline; devices upload anonymous results here when internet is available."""
from __future__ import annotations
import csv
import hmac
import io
import json
import os
import re
import secrets
import shutil
import sqlite3
import tempfile
import threading
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from urllib.parse import quote
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from . import ingest, pack as packlib
from .library import import_book, import_folder, infer_meta
from .report import build_pdf
from .retriever import BM25Index
from .tutor import Tutor

WEB = Path(__file__).parent / "web"
DAY = 86400
# dashboard dimension -> column (a whitelist, so user input never reaches the SQL text)
DIMS = {"subject": "subject", "school": "school_id", "project": "project_id",
        "class": "class_id", "topic": "topic"}


class Event(BaseModel):
    topic: str
    question_id: str
    correct: bool
    ts: int = Field(description="unix seconds on the device")
    subject: str = ""


class SyncRequest(BaseModel):
    device_id: str  # random UUID = one anonymous student/device, never a name
    class_id: str
    school_id: str = ""
    project_id: str = ""
    events: list[Event]


class Login(BaseModel):
    password: str


class AskRequest(BaseModel):
    question: str
    subject: str | None = None
    grade: int | None = None
    lang: str = "English"  # language the AI model should answer in (ignored without a model)


class Filters:
    """Query-string filters shared by every analytics endpoint."""

    def __init__(self, school: str = "", project: str = "", class_id: str = "",
                 subject: str = "", days: int = Query(0, ge=0)):
        self.school, self.project, self.class_id, self.subject, self.days = school, project, class_id, subject, days


def create_app(db_path: str = "aitutor.db", content_version: int = 1, passages_path: str | None = None,
               books_dir: str | None = None, clock=time.time, admin_password: str | None = None,
               sync_key: str | None = None, secret: str | None = None, pack_path: str | None = None,
               backend=None, model_path: str | None = None, gpu_layers: int = 0,
               student_code: str | None = None) -> FastAPI:
    app = FastAPI(title="AI Textbook Tutor", version="0.6.0")
    pp = Path(passages_path or os.environ.get("AITUTOR_PASSAGES", "data/passages.jsonl"))
    bdir = Path(books_dir or os.environ.get("AITUTOR_BOOKS", "books"))
    if backend is None and model_path:
        from .tutor import LlamaCppBackend
        backend = LlamaCppBackend(model_path, n_gpu_layers=gpu_layers)
    model_lock = threading.Lock()  # llama.cpp runs one generation at a time

    class Lib:
        """The imported books. Rebuilt in place when a teacher uploads or deletes a book."""

        def __init__(self):
            self.set(ingest.load(pp) if pp.exists() else [])

        def set(self, passages):
            self.passages = passages
            self.by_id = {p.id: p for p in passages}
            self.known_files = {p.source for p in passages if p.source}
            self.tutor = Tutor(BM25Index(passages), backend)

        def reload(self):  # rebuild everything from the books folder and save it for the next start
            ps, _ = import_folder(bdir) if bdir.is_dir() else ([], [])
            pp.parent.mkdir(parents=True, exist_ok=True)
            ingest.save(ps, pp)
            self.set(ps)

    lib = Lib()
    if not pp.exists() and bdir.is_dir() and any(f.suffix.lower() in (".pdf", ".epub") for f in bdir.iterdir()):
        lib.reload()  # first start with books but no imported data: import them now

    # Students need no login (the tutor runs on their device). The dashboard needs the admin password;
    # uploads need the sync key if one is set.
    admin_pw = admin_password if admin_password is not None else os.environ.get("AITUTOR_ADMIN_PASSWORD")
    skey = sync_key if sync_key is not None else os.environ.get("AITUTOR_SYNC_KEY")
    sec = (secret or os.environ.get("AITUTOR_SECRET") or secrets.token_hex(32)).encode()
    pk = pack_path or os.environ.get("AITUTOR_PACK")
    scode = student_code if student_code is not None else os.environ.get("AITUTOR_STUDENT_CODE")
    fails: dict[str, list[float]] = {}
    COOKIE, TTL = "aitutor_session", 12 * 3600
    SCOOKIE, STTL = "aitutor_student", 7 * 24 * 3600

    def eq(a: str, b: str) -> bool:
        return hmac.compare_digest(a.encode(), b.encode())

    def sign(exp: int, role: str = "a") -> str:  # role keeps admin and student cookies from being swapped
        return f"{exp}.{hmac.new(sec, f'{role}:{exp}'.encode(), 'sha256').hexdigest()}"

    def session_ok(tok: str | None, role: str = "a") -> bool:
        try:
            exp, _ = (tok or "").split(".", 1)
            return int(exp) > clock() and eq(tok, sign(int(exp), role))
        except ValueError:
            return False

    def is_admin(request: Request) -> bool:
        if not admin_pw:
            return False
        bearer = request.headers.get("authorization", "")
        return session_ok(request.cookies.get(COOKIE)) or (bearer.startswith("Bearer ") and eq(bearer[7:], admin_pw))

    def require_admin(request: Request):
        if not admin_pw:
            raise HTTPException(403, "Dashboard is disabled: set AITUTOR_ADMIN_PASSWORD")
        if not is_admin(request):
            raise HTTPException(401, "Login required")

    def require_key(key: str | None):
        if skey and not eq(key or "", skey):
            raise HTTPException(401, "Missing or wrong sync key")

    ADMIN = Depends(require_admin)

    def student_ok(request: Request) -> bool:
        """With an access code set, students must enter it (admins are always allowed). No code = open."""
        if not scode:
            return True
        return (is_admin(request) or session_ok(request.cookies.get(SCOOKIE), "s")
                or eq(request.headers.get("x-student-code", ""), scode))

    def require_student(request: Request):
        if not student_ok(request):
            raise HTTPException(401, "Access code required")

    STUDENT = Depends(require_student)

    def db() -> sqlite3.Connection:
        con = sqlite3.connect(db_path)
        con.execute("""CREATE TABLE IF NOT EXISTS events(
            device_id TEXT, class_id TEXT, topic TEXT, question_id TEXT,
            correct INTEGER, ts INTEGER, UNIQUE(device_id, question_id, ts))""")
        have = {r[1] for r in con.execute("PRAGMA table_info(events)")}
        for col in ("school_id", "project_id", "subject"):  # migrate databases from older versions
            if col not in have:
                con.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT DEFAULT ''")
        return con

    def rows(sql: str, params: tuple = ()) -> list[tuple]:
        with closing(db()) as con:
            return con.execute(sql, params).fetchall()

    def where(f: Filters, extra: str = "", more: tuple = ()) -> tuple[str, tuple]:
        clause, params = "WHERE 1=1", []
        for col, val in (("school_id", f.school), ("project_id", f.project),
                         ("class_id", f.class_id), ("subject", f.subject)):
            if val:
                clause += f" AND {col}=?"; params.append(val)
        if f.days:
            clause += " AND ts>=?"; params.append(int(clock()) - f.days * DAY)
        return clause + extra, tuple(params) + more

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/content/version")
    def version():
        if pk and Path(pk).is_file():  # devices pull the content pack when this is higher than theirs
            return {"version": packlib.read_manifest(Path(pk))["version"]}
        return {"version": content_version}

    @app.get("/content/pack")
    def content_pack(x_sync_key: str | None = Header(None)):
        require_key(x_sync_key)  # books may be copyrighted: same key as uploads
        if not pk or not Path(pk).is_file():
            raise HTTPException(404, "no content pack is published on this server")
        return FileResponse(pk, media_type="application/zip", filename="tutor.pack")

    @app.post("/api/login")
    def login(body: Login, request: Request, response: Response):
        if not admin_pw:
            raise HTTPException(403, "Dashboard is disabled: set AITUTOR_ADMIN_PASSWORD")
        ip, now = (request.client.host if request.client else "?"), clock()
        recent = [t for t in fails.get(ip, []) if now - t < 300]
        if len(recent) >= 5:
            raise HTTPException(429, "Too many wrong passwords. Wait a few minutes.")
        if not eq(body.password, admin_pw):
            fails[ip] = recent + [now]
            raise HTTPException(401, "Wrong password")
        fails.pop(ip, None)
        response.set_cookie(COOKIE, sign(int(now) + TTL), max_age=TTL, httponly=True, samesite="lax",
                            secure=request.url.scheme == "https")
        return {"ok": True}

    @app.post("/api/enter")
    def enter(body: dict, request: Request, response: Response):
        if not scode:
            return {"ok": True}
        ip, now = "s:" + (request.client.host if request.client else "?"), clock()
        recent = [t for t in fails.get(ip, []) if now - t < 300]
        if len(recent) >= 5:
            raise HTTPException(429, "Too many wrong codes. Wait a few minutes.")
        if not eq(str(body.get("code", "")), scode):
            fails[ip] = recent + [now]
            raise HTTPException(401, "Wrong access code")
        fails.pop(ip, None)
        response.set_cookie(SCOOKIE, sign(int(now) + STTL, "s"), max_age=STTL, httponly=True, samesite="lax",
                            secure=request.url.scheme == "https")
        return {"ok": True}

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE)
        return {"ok": True}

    @app.post("/sync")
    def sync(req: SyncRequest, x_sync_key: str | None = Header(None)):
        require_key(x_sync_key)
        added = 0
        with closing(db()) as con:
            for e in req.events:  # INSERT OR IGNORE makes retries idempotent
                cur = con.execute(
                    """INSERT OR IGNORE INTO events(device_id, class_id, school_id, project_id, subject,
                       topic, question_id, correct, ts) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (req.device_id, req.class_id, req.school_id, req.project_id, e.subject,
                     e.topic, e.question_id, int(e.correct), e.ts))
                added += cur.rowcount
            con.commit()
        return {"accepted": added, "duplicates": len(req.events) - added}

    @app.get("/dashboard/{class_id}", dependencies=[ADMIN])
    def dashboard(class_id: str, min_attempts: int = 5):
        f = Filters(class_id=class_id, days=0)
        w, p = where(f)
        topics = rows(f"SELECT topic, COUNT(*), AVG(correct), COUNT(DISTINCT device_id) FROM events {w} "
                      "GROUP BY topic ORDER BY AVG(correct)", p)
        hard = rows(f"SELECT question_id, topic, COUNT(*), AVG(correct) FROM events {w} "
                    "GROUP BY question_id HAVING COUNT(*)>=? ORDER BY AVG(correct) LIMIT 3", p + (min_attempts,))
        return {
            "class_id": class_id,
            "topics": [{"topic": t, "attempts": n, "accuracy": round(a, 2), "students": s,
                        "needs_review": n >= min_attempts and a < 0.5} for t, n, a, s in topics],
            "hardest_questions": [{"question_id": q, "topic": t, "attempts": n, "accuracy": round(a, 2)}
                                  for q, t, n, a in hard]}

    @app.get("/analytics/filters", dependencies=[ADMIN])
    def filter_options():
        pick = lambda col: [r[0] for r in rows(f"SELECT DISTINCT {col} FROM events WHERE {col}<>'' ORDER BY 1")]
        return {"school": pick("school_id"), "project": pick("project_id"),
                "class_id": pick("class_id"), "subject": pick("subject")}

    @app.get("/analytics/summary", dependencies=[ADMIN])
    def summary(f: Filters = Depends()):
        w, p = where(f)
        s, n, a, sch, cls, sub = rows(
            "SELECT COUNT(DISTINCT device_id), COUNT(*), AVG(correct), COUNT(DISTINCT NULLIF(school_id,'')), "
            f"COUNT(DISTINCT NULLIF(class_id,'')), COUNT(DISTINCT NULLIF(subject,'')) FROM events {w}", p)[0]
        w7, p7 = where(f, " AND ts>=?", (int(clock()) - 7 * DAY,))
        active = rows(f"SELECT COUNT(DISTINCT device_id) FROM events {w7}", p7)[0][0]
        return {"students": s, "active_7d": active, "attempts": n, "accuracy": round(a or 0, 2),
                "schools": sch, "classes": cls, "subjects": sub}

    @app.get("/analytics/by/{dim}", dependencies=[ADMIN])
    def by_dimension(dim: str, f: Filters = Depends(), min_attempts: int = 5):
        if dim not in DIMS:
            raise HTTPException(422, f"dim must be one of {sorted(DIMS)}")
        col, (w, p) = DIMS[dim], where(f)
        data = rows(f"SELECT {col}, COUNT(DISTINCT device_id), COUNT(*), AVG(correct) FROM events {w} "
                    f"GROUP BY {col} ORDER BY AVG(correct)", p)
        return {"dim": dim, "rows": [{"name": name or "(none)", "students": s, "attempts": n,
                                      "accuracy": round(a, 2), "needs_review": n >= min_attempts and a < 0.5}
                                     for name, s, n, a in data]}

    @app.get("/analytics/timeseries", dependencies=[ADMIN])
    def timeseries(f: Filters = Depends()):
        w, p = where(f)
        data = rows("SELECT date(ts,'unixepoch'), COUNT(*), AVG(correct), COUNT(DISTINCT device_id) "
                    f"FROM events {w} GROUP BY 1 ORDER BY 1", p)
        return {"days": [{"day": d, "attempts": n, "accuracy": round(a, 2), "students": s}
                         for d, n, a, s in data]}

    @app.get("/analytics/students", dependencies=[ADMIN])
    def students(f: Filters = Depends(), limit: int = Query(50, ge=1, le=500)):
        w, p = where(f)
        data = rows("SELECT device_id, MAX(class_id), MAX(school_id), COUNT(*), AVG(correct), MAX(ts) "
                    f"FROM events {w} GROUP BY device_id", p)
        stale, out = int(clock()) - 14 * DAY, []
        for dev, cls, sch, n, a, last in data:
            reasons = []
            if n >= 5 and a < 0.4:
                reasons.append("low accuracy")
            if last < stale:
                reasons.append("inactive 14+ days")
            out.append({"student": dev[:8], "class": cls, "school": sch, "attempts": n,
                        "accuracy": round(a, 2), "last_active": time.strftime("%Y-%m-%d", time.gmtime(last)),
                        "at_risk": reasons})
        out.sort(key=lambda r: (not r["at_risk"], r["accuracy"]))
        return {"total": len(out), "at_risk_count": sum(1 for r in out if r["at_risk"]), "students": out[:limit]}

    @app.get("/analytics/questions", dependencies=[ADMIN])
    def hardest(f: Filters = Depends(), min_attempts: int = 3):
        w, p = where(f)
        data = rows("SELECT question_id, MAX(topic), MAX(subject), COUNT(*), AVG(correct) FROM events "
                    f"{w} GROUP BY question_id HAVING COUNT(*)>=? ORDER BY AVG(correct), COUNT(*) DESC LIMIT 10",
                    p + (min_attempts,))
        return {"questions": [{"question_id": q, "topic": t, "subject": s, "attempts": n, "accuracy": round(a, 2)}
                              for q, t, s, n, a in data]}

    @app.get("/analytics/export.csv", dependencies=[ADMIN])
    def export(dim: str = "school", f: Filters = Depends()):
        table = by_dimension(dim, f)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([dim, "students", "attempts", "accuracy", "needs_review"])
        for r in table["rows"]:
            w.writerow([r["name"], r["students"], r["attempts"], r["accuracy"], r["needs_review"]])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="aitutor_by_{dim}.csv"'})

    @app.get("/analytics/report.pdf", dependencies=[ADMIN])
    def report(f: Filters = Depends()):
        s = summary(f)
        by = lambda d: by_dimension(d, f)["rows"]
        fmt = lambda rows_: [[r["name"], r["students"], r["attempts"], f"{round(r['accuracy'] * 100)}%",
                              "review" if r["needs_review"] else ""] for r in rows_]
        head = ["", "Students", "Attempts", "Accuracy", ""]
        risk = students(f, limit=50)
        scope = " | ".join(x for x in (f.school, f.project, f.class_id, f.subject) if x) or "All schools"
        period = f"last {f.days} days" if f.days else "all time"
        pdf = build_pdf(
            f"AI Tutor report: {scope}",
            f"{time.strftime('%Y-%m-%d', time.gmtime(clock()))} UTC, period: {period}. Students = anonymous devices.",
            [("Students", str(s["students"])), ("Active 7 days", str(s["active_7d"])),
             ("Attempts", str(s["attempts"])), ("Accuracy", f"{round(s['accuracy'] * 100)}%"),
             ("Need attention", str(risk["at_risk_count"]))],
            [("By class", ["Class"] + head[1:], fmt(by("class"))), ("By subject", ["Subject"] + head[1:], fmt(by("subject"))),
             ("Weakest topics", ["Topic"] + head[1:], fmt(by("topic"))[:10])])
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="ai_tutor_report.pdf"'})

    @app.get("/api/library", dependencies=[STUDENT])
    def library():
        books: dict[str, dict] = {}
        for p in lib.passages:
            b = books.setdefault(p.book or p.unit, {"book": p.unit, "subject": p.subject, "grade": p.grade,
                                                    "lang": p.lang, "source": p.source, "passages": 0})
            b["passages"] += 1
        return {"ai": backend is not None, "books": list(books.values()), "subjects": sorted({p.subject for p in lib.passages}),
                "grades": sorted({p.grade for p in lib.passages if p.grade})}

    @app.post("/api/ask", dependencies=[STUDENT])
    def ask(req: AskRequest):
        with model_lock:
            r = lib.tutor.ask(req.question, grade=req.grade, subject=req.subject, lang=req.lang)
        cites = []
        for p in r.passages:
            c = {"text": p.cite(), "passage_id": p.id, "page": p.page, "pdf_page": p.phys, "excerpt": p.text,
                 "link": None, "preview": None}
            if p.source.lower().endswith(".pdf") and p.phys and (bdir / p.source).is_file():
                c["link"] = f"/files/{quote(p.source)}#page={p.phys}"  # opens the PDF at that page
                c["preview"] = f"/preview/{p.id}.png?q={quote(req.question)}"
            cites.append(c)
        return {"grounded": r.grounded, "hint": r.hint, "answer": r.answer, "citations": cites,
                "ai": backend is not None and r.grounded, "removed": r.removed}

    @app.get("/files/{name}", dependencies=[STUDENT])
    def book_file(name: str):
        f = bdir / name
        if name not in lib.known_files or Path(name).name != name or not f.is_file():  # no path tricks
            raise HTTPException(404, "unknown book")
        media = "application/pdf" if f.suffix.lower() == ".pdf" else "application/octet-stream"
        return FileResponse(f, media_type=media)

    @app.get("/preview/{pid}.png", dependencies=[STUDENT])
    def preview(pid: str, q: str = ""):
        p = lib.by_id.get(pid)
        if not p or not p.source.lower().endswith(".pdf") or not p.phys or not (bdir / p.source).is_file():
            raise HTTPException(404, "no preview for this source")
        try:
            from .preview import render_page
        except ImportError:
            raise HTTPException(501, "Install pypdfium2 and pillow to enable page previews")
        return Response(render_page(bdir / p.source, p.phys, q), media_type="image/png",
                        headers={"Cache-Control": "max-age=3600"})

    max_mb = int(os.environ.get("AITUTOR_MAX_UPLOAD_MB", "150"))
    name_ok = re.compile(r"^\w[\w .()\-]*\.(pdf|epub)$", re.I)
    langs = {"en", "am", "om", "so", "ti"}

    def manifest() -> dict:
        f = bdir / "manifest.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    def save_manifest(m: dict) -> None:
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")

    @app.get("/admin/books", dependencies=[ADMIN])
    def list_books():
        counts, meta, out = Counter(p.source for p in lib.passages), manifest(), []
        for f in sorted(bdir.iterdir()) if bdir.is_dir() else []:
            if f.suffix.lower() in (".pdf", ".epub"):
                m = {**infer_meta(f.stem), **meta.get(f.name, {})}
                out.append({"file": f.name, "title": m.get("title", ""), "subject": m.get("subject", "general"),
                            "grade": m.get("grade", 0), "lang": m.get("lang", "en"),
                            "passages": counts.get(f.name, 0), "size_mb": round(f.stat().st_size / 1e6, 2)})
        return {"books": out, "max_upload_mb": max_mb}

    @app.put("/admin/books/{name}", dependencies=[ADMIN])
    async def upload_book(name: str, request: Request, subject: str = "general",
                          grade: int = Query(0, ge=0, le=12), lang: str = "en", title: str = ""):
        if not name_ok.match(name) or Path(name).name != name:
            raise HTTPException(400, "File name must end in .pdf or .epub and use only letters, numbers, spaces, - _ . ( )")
        if lang not in langs:
            raise HTTPException(400, f"lang must be one of {sorted(langs)}")
        if int(request.headers.get("content-length") or 0) > max_mb * 1_000_000:
            raise HTTPException(413, f"File is larger than {max_mb} MB")
        data = await request.body()
        if len(data) > max_mb * 1_000_000:
            raise HTTPException(413, f"File is larger than {max_mb} MB")
        subject = re.sub(r"[^a-z0-9]+", "", subject.lower()) or "general"

        def validate_and_store() -> int:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / name
                tmp.write_bytes(data)
                try:  # read it first: a bad file must never replace a good library
                    found = import_book(tmp, subject, grade, lang, title or None)
                except Exception as e:
                    raise HTTPException(422, f"Could not read this file ({type(e).__name__}). "
                                             "Is it a valid, unencrypted PDF or EPUB?")
                if not found:
                    raise HTTPException(422, "No readable text found. Scanned books need OCR first.")
                bdir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tmp, bdir / name)
            m = manifest()
            m[name] = {"subject": subject, "grade": grade, "lang": lang, **({"title": title} if title else {})}
            save_manifest(m)
            lib.reload()
            return len(found)

        n = await run_in_threadpool(validate_and_store)
        return {"file": name, "passages": n, "subject": subject, "grade": grade, "lang": lang}

    @app.delete("/admin/books/{name}", dependencies=[ADMIN])
    def delete_book(name: str):
        f = bdir / name
        if not name_ok.match(name) or Path(name).name != name or not f.is_file():
            raise HTTPException(404, "no such book")
        f.unlink()
        m = manifest()
        m.pop(name, None)
        save_manifest(m)
        lib.reload()
        return {"deleted": name}

    @app.post("/admin/reindex", dependencies=[ADMIN])
    def reindex():
        lib.reload()
        return {"passages": len(lib.passages)}

    @app.get("/", response_class=HTMLResponse)
    def student_page(request: Request):
        if not student_ok(request):
            return RedirectResponse("/enter", status_code=303)
        return (WEB / "index.html").read_text(encoding="utf-8")

    @app.get("/teacher", response_class=HTMLResponse)
    def teacher_page(request: Request):
        if not is_admin(request):
            return RedirectResponse("/login", status_code=303)
        return (WEB / "teacher.html").read_text(encoding="utf-8")

    @app.get("/teacher/library", response_class=HTMLResponse)
    def library_page(request: Request):
        if not is_admin(request):
            return RedirectResponse("/login", status_code=303)
        return (WEB / "library.html").read_text(encoding="utf-8")

    @app.get("/enter", response_class=HTMLResponse)
    def enter_page():
        return (WEB / "enter.html").read_text(encoding="utf-8")

    @app.get("/login", response_class=HTMLResponse)
    def login_page():
        return (WEB / "login.html").read_text(encoding="utf-8")

    return app


app = create_app()
