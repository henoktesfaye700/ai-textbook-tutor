"""Commands:
  aitutor import books/ --out data/passages.jsonl      import a folder of PDFs/EPUBs
  aitutor books --passages data/passages.jsonl         list imported books
  aitutor ask "question" --subject biology --grade 8   grounded answer with citations
  aitutor quiz --subject biology --grade 8 -n 5        adaptive practice (saves outbox)
  aitutor sync --url http://host:8000 --class 8A --school "Hawassa" --project "Pilot"
  aitutor serve [--admin-password P] [--sync-key K]    web app: / (students), /teacher (dashboard)
  aitutor pack build|install|pull                      offline content packs (SD card / USB / download)
"""
from __future__ import annotations
import argparse
import json
import secrets
import tempfile
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path
from . import ingest, pack as packlib
from .library import import_folder
from .quiz import QuizSession, generate, save_outbox
from .retriever import BM25Index
from .tutor import LlamaCppBackend, Tutor


def _load(args):
    ps = ingest.load(Path(args.passages))
    return [p for p in ps if (not args.subject or p.subject == args.subject.lower())
            and (not args.grade or p.grade in (0, args.grade))]


def cmd_ask(a):
    index = BM25Index(ingest.load(Path(a.passages)))
    backend = LlamaCppBackend(a.model, n_gpu_layers=a.gpu_layers) if a.model else None
    r = Tutor(index, backend).ask(a.question, grade=a.grade, subject=a.subject, lang=a.lang)
    if not r.grounded:
        print(r.answer)
        return
    print("THINK FIRST:", r.hint, "\n\nANSWER:", r.answer, "\n\nSOURCES:", *r.citations, sep="\n")
    if backend:
        print(f"\n(AI answer checked against the textbook; {r.removed} unsupported sentence(s) removed)")


def cmd_quiz(a):
    ps = _load(a)
    qs = generate(ps, BM25Index(ps))[: a.n]
    if not qs:
        print("Not enough text to build questions. Import more books.")
        return
    s = QuizSession(qs)
    while (q := s.next()):
        print(f"\n[{q.topic}]  {q.prompt}")
        for i, o in enumerate(q.options):
            print(f"  {i + 1}. {o}")
        try:
            choice = int(input("Your answer (number): ")) - 1
        except (ValueError, EOFError):
            break
        ok = s.answer(q, choice)
        print("Correct." if ok else f"Not yet. Answer: {q.options[q.answer]}  (see {q.source})")
    print("\nMastery estimates:", s.mastery())
    out = Path(a.outbox)
    dev = json.loads(out.read_text())["device_id"] if out.exists() else str(uuid.uuid4())
    save_outbox(out, s.events, dev)
    print(f"{len(s.events)} results saved to {out} (upload later with `aitutor sync`).")


def cmd_sync(a):
    out = Path(a.outbox)
    data = json.loads(out.read_text())
    body = json.dumps({"device_id": data["device_id"], "class_id": a.cls, "school_id": a.school,
                       "project_id": a.project, "events": data["events"]}).encode()
    headers = {"Content-Type": "application/json", **({"X-Sync-Key": a.sync_key} if a.sync_key else {})}
    req = urllib.request.Request(a.url.rstrip("/") + "/sync", body, headers)
    try:
        print(json.load(urllib.request.urlopen(req, timeout=10)))
    except OSError as e:
        print("Offline or server unreachable, outbox kept:", e)
        return
    data["events"] = []
    out.write_text(json.dumps(data))


def cmd_pack(a):
    if a.action == "build":
        m = packlib.build(Path(a.passages), Path(a.books), Path(a.out), a.version, a.sign_key, not a.no_books)
        print(f"built {a.out}: version {m['version']}, {m['passages']} passages, {len(m['files']) - 1} books, "
              f"{'signed' if m['signature'] else 'unsigned'}")
    elif a.action == "install":
        m = packlib.install(Path(a.file), Path(a.data), Path(a.books), a.sign_key, a.force)
        print(f"installed version {m['version']} ({m['passages']} passages). Restart `aitutor serve` to use it.")
    else:  # pull: download the server's pack if it is newer than the installed one
        base, hdr = a.url.rstrip("/"), ({"X-Sync-Key": a.sync_key} if a.sync_key else {})
        marker = Path(a.data) / "installed_pack.json"
        have = json.loads(marker.read_text())["version"] if marker.exists() else 0
        try:
            latest = json.load(urllib.request.urlopen(base + "/content/version", timeout=10))["version"]
            if latest <= have:
                print(f"already up to date (version {have})")
                return
            with tempfile.NamedTemporaryFile(suffix=".tutorpack", delete=False) as tmp:
                tmp.write(urllib.request.urlopen(urllib.request.Request(base + "/content/pack", headers=hdr), timeout=60).read())
        except OSError as e:
            print("Offline or server unreachable, nothing changed:", e)
            return
        a.file = tmp.name
        a.action = "install"
        cmd_pack(a)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aitutor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--passages", default="data/passages.jsonl")
    common.add_argument("--subject")
    common.add_argument("--grade", type=int)

    i = sub.add_parser("ingest", help="legacy marked-up text file"); i.add_argument("raw"); i.add_argument("out")
    m = sub.add_parser("import"); m.add_argument("folder"); m.add_argument("--out", default="data/passages.jsonl")
    sub.add_parser("books", parents=[common])
    a = sub.add_parser("ask", parents=[common]); a.add_argument("question")
    a.add_argument("--model", help="optional GGUF model for on-device LLM")
    a.add_argument("--lang", default="English", help="language for the AI answer")
    a.add_argument("--gpu-layers", type=int, default=0, help="layers to offload to a GPU (0 = CPU only)")
    q = sub.add_parser("quiz", parents=[common]); q.add_argument("-n", type=int, default=5)
    q.add_argument("--outbox", default="outbox.json")
    s = sub.add_parser("sync"); s.add_argument("--url", required=True)
    s.add_argument("--class", dest="cls", required=True); s.add_argument("--outbox", default="outbox.json")
    s.add_argument("--school", default=""); s.add_argument("--project", default="")
    s.add_argument("--sync-key", default=None, help="shared key if the server requires one")
    v = sub.add_parser("serve", help="student page at /, teacher dashboard at /teacher")
    v.add_argument("--passages", default="data/passages.jsonl"); v.add_argument("--books", default="books")
    v.add_argument("--db", default="aitutor.db"); v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--admin-password", default=None, help="dashboard password (generated if omitted)")
    v.add_argument("--sync-key", default=None, help="require this key for uploads and pack downloads")
    v.add_argument("--student-code", default=None, help="access code students must enter (recommended on a public site)")
    v.add_argument("--pack", default=None, help="content pack to publish to other devices")
    v.add_argument("--model", default=None, help="GGUF model file: students get AI-written, textbook-checked answers")
    v.add_argument("--gpu-layers", type=int, default=0)
    v.add_argument("--port", type=int, default=int(__import__("os").environ.get("PORT", 8000)),
                   help="default: $PORT if set (most web hosts set it), else 8000")
    k = sub.add_parser("pack", help="build / install / pull an offline content pack")
    k.add_argument("action", choices=["build", "install", "pull"])
    k.add_argument("file", nargs="?", help="pack file (install)")
    k.add_argument("--passages", default="data/passages.jsonl"); k.add_argument("--books", default="books")
    k.add_argument("--data", default="data"); k.add_argument("--out", default="school.tutorpack")
    k.add_argument("--version", type=int, default=1); k.add_argument("--no-books", action="store_true")
    k.add_argument("--sign-key", default=None, help="sign (build) or verify (install/pull) with this key")
    k.add_argument("--sync-key", default=None); k.add_argument("--url"); k.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "ingest":
        ps = ingest.parse_raw(Path(args.raw).read_text(encoding="utf-8"))
        ingest.save(ps, Path(args.out)); print(f"wrote {len(ps)} passages to {args.out}")
    elif args.cmd == "import":
        ps, report = import_folder(Path(args.folder))
        ingest.save(ps, Path(args.out))
        for r in report:
            print(r)
        print(f"wrote {len(ps)} passages to {args.out}")
    elif args.cmd == "books":
        g = defaultdict(list)
        for p in ingest.load(Path(args.passages)):
            g[(p.book or p.unit, p.subject, p.grade, p.lang)].append(p)
        for (b, sj, gr, lg), ps in g.items():
            print(f"{b}: subject={sj} grade={gr} lang={lg} passages={len(ps)} chapters={len({p.topic for p in ps})}")
    elif args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "quiz":
        cmd_quiz(args)
    elif args.cmd == "sync":
        cmd_sync(args)
    elif args.cmd == "serve":
        import os
        import uvicorn
        from .api import create_app
        pw = args.admin_password or os.environ.get("AITUTOR_ADMIN_PASSWORD")
        if not pw:  # secure by default: never run the dashboard without a password
            pw = secrets.token_urlsafe(9)
            print(f"Generated dashboard password (shown once): {pw}")
        print(f"Students: http://{args.host}:{args.port}/   Teachers: http://{args.host}:{args.port}/teacher")
        uvicorn.run(create_app(args.db, passages_path=args.passages, books_dir=args.books, admin_password=pw,
                               sync_key=args.sync_key, pack_path=args.pack, model_path=args.model,
                               gpu_layers=args.gpu_layers, student_code=args.student_code), host=args.host, port=args.port)
    elif args.cmd == "pack":
        try:
            cmd_pack(args)
        except packlib.PackError as e:
            raise SystemExit(f"Pack rejected, nothing was changed: {e}")


if __name__ == "__main__":
    main()
