"""Content packs: one file holding the passages and book PDFs, for schools without internet.
Copy it by SD card, USB or Bluetooth, then `aitutor pack install`. Integrity is checked with SHA-256
checksums; an optional shared key adds a signature so a tampered pack is rejected."""
from __future__ import annotations
import hashlib
import hmac
import json
import time
import zipfile
from pathlib import Path

FORMAT = 1


class PackError(Exception):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sign(files: dict[str, str], key: str) -> str:
    return hmac.new(key.encode(), json.dumps(files, sort_keys=True).encode(), "sha256").hexdigest()


def read_manifest(pack: Path) -> dict:
    with zipfile.ZipFile(pack) as z:
        return json.loads(z.read("manifest.json"))


def build(passages: Path, books_dir: Path | None, out: Path, version: int,
          key: str | None = None, include_books: bool = True) -> dict:
    data = passages.read_bytes()
    entries = {"passages.jsonl": data}
    sources = {json.loads(l)["source"] for l in data.decode("utf-8").splitlines() if l.strip()} - {""}
    if include_books and books_dir:
        for name in sorted(sources):
            f = books_dir / name
            if f.is_file():
                entries[f"books/{name}"] = f.read_bytes()
    files = {n: _sha(b) for n, b in entries.items()}
    manifest = {"format": FORMAT, "version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "passages": sum(1 for l in data.splitlines() if l.strip()), "files": files,
                "signature": _sign(files, key) if key else None}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        for n, b in entries.items():
            z.writestr(n, b)
    return manifest


def install(pack: Path, data_dir: Path, books_dir: Path, key: str | None = None, force: bool = False) -> dict:
    with zipfile.ZipFile(pack) as z:
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise PackError(f"unsupported pack format: {manifest.get('format')}")
        files = manifest["files"]
        if key:
            sig = manifest.get("signature")
            if not sig or not hmac.compare_digest(sig, _sign(files, key)):
                raise PackError("signature missing or wrong: pack not trusted")
        if set(z.namelist()) - {"manifest.json"} != set(files):
            raise PackError("pack contains files that are not listed in its manifest")
        blobs = {}
        for name, digest in files.items():
            if name.startswith("/") or ".." in Path(name).parts:
                raise PackError(f"unsafe path in pack: {name}")
            blobs[name] = z.read(name)
            if _sha(blobs[name]) != digest:
                raise PackError(f"checksum mismatch: {name} (file damaged or altered)")
    marker = data_dir / "installed_pack.json"
    if marker.exists() and not force and manifest["version"] <= json.loads(marker.read_text())["version"]:
        raise PackError(f"version {manifest['version']} is not newer than the installed one")
    data_dir.mkdir(parents=True, exist_ok=True)
    books_dir.mkdir(parents=True, exist_ok=True)
    for name, b in blobs.items():  # everything verified first, then written
        dest = data_dir / name if name == "passages.jsonl" else books_dir / Path(name).name
        dest.write_bytes(b)
    marker.write_text(json.dumps({"version": manifest["version"], "created": manifest["created"]}))
    return manifest
