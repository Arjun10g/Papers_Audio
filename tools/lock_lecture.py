#!/usr/bin/env python3
"""Publish a password-protected lecture: only ciphertext ever leaves this machine.

    LECTURE_PASSWORD=… python tools/lock_lecture.py <id>      # or omit the variable to be prompted

Inputs (all git-ignored, never published in the clear):
    private/lectures/<id>.md      the narration text (also the in-app transcript)
    private/catalog.json          {"lectures": [{"id", "title", "series", "bg", ...}]}
    audio/<id>.mp3                rendered with tools/generate_lecture.py <id>
    audio/<id>.chapters.json

What it does:
1. Derives a 256-bit key from the password: PBKDF2-HMAC-SHA-256, 600,000 iterations, random
   16-byte salt.
2. Encrypts the MP3 and the transcript with AES-256-GCM (random 12-byte IV, stored as
   IV || ciphertext || tag) and uploads them to the public dataset as locked/<opaque>.mp3.enc and
   locked/<opaque>.md.enc, in one commit.
3. Encrypts the title, series, artwork and chapter list (plus the transcript URL) into the
   library entry, and writes it to locked.json. That file is safe to commit: its visible fields
   are placeholders ("Locked lecture"); only duration and size are real. tools/publish.py
   appends locked.json to library.json.
4. Decrypts everything again to prove the round trip before finishing.

The opaque id for each lecture is remembered in private/locked_ids.json, so re-locking (for
example with a new password) replaces the same entry. The app keeps a remembered key per
lecture id and salt, so a new password invalidates old remembered keys.

The strength of the protection is the strength of the password: the ciphertext is public, so a
short password can be guessed offline.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import getpass
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "private"
LOCKED_JSON = ROOT / "locked.json"
REPO_ID, REPO_TYPE, HUB = "arjun10g/papers-audio", "dataset", "https://huggingface.co"
ITERATIONS = 600_000

b64 = lambda b: base64.b64encode(b).decode("ascii")


def derive(password: str, salt: bytes, iterations: int = ITERATIONS) -> bytes:
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations).derive(
        password.encode("utf-8"))


def seal(key: bytes, plain: bytes) -> bytes:
    iv = os.urandom(12)
    return iv + AESGCM(key).encrypt(iv, plain, None)          # ciphertext || 16-byte tag


def unseal(key: bytes, blob: bytes) -> bytes:
    return AESGCM(key).decrypt(blob[:12], blob[12:], None)


def duration_of(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return round(float(out), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or None)
    args = ap.parse_args()
    lid = args.id

    cat = json.loads((PRIVATE / "catalog.json").read_text(encoding="utf-8"))
    entry = next((l for l in cat["lectures"] if l["id"] == lid), None)
    if not entry:
        sys.exit(f"{lid} is not in private/catalog.json")
    mp3, chapters_f, doc_f = ROOT / "audio" / f"{lid}.mp3", ROOT / "audio" / f"{lid}.chapters.json", \
        PRIVATE / "lectures" / f"{lid}.md"
    for f in (mp3, chapters_f, doc_f):
        if not f.exists():
            sys.exit(f"missing {f.relative_to(ROOT)}")

    password = os.environ.get("LECTURE_PASSWORD") or getpass.getpass(f"Password for {lid}: ")
    if not password:
        sys.exit("empty password")

    ids_f = PRIVATE / "locked_ids.json"
    ids = json.loads(ids_f.read_text()) if ids_f.exists() else {}
    opaque = ids.setdefault(lid, secrets.token_hex(4))
    ids_f.write_text(json.dumps(ids, indent=1) + "\n")

    salt = os.urandom(16)
    key = derive(password, salt)
    audio_ct = seal(key, mp3.read_bytes())
    doc_ct = seal(key, doc_f.read_bytes())

    # 1. ciphertext files, one commit (the URLs below are pinned to it)
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=args.token)
    audio_path, doc_path = f"locked/{opaque}.mp3.enc", f"locked/{opaque}.md.enc"
    info = api.create_commit(REPO_ID, repo_type=REPO_TYPE, commit_message=f"locked lecture {opaque}", operations=[
        CommitOperationAdd(path_in_repo=audio_path, path_or_fileobj=audio_ct),
        CommitOperationAdd(path_in_repo=doc_path, path_or_fileobj=doc_ct)])
    url = lambda p: f"{HUB}/datasets/{REPO_ID}/resolve/{info.oid}/{p}"
    print(f"uploaded {len(audio_ct) / 1e6:.1f} MB audio + {len(doc_ct) / 1e3:.0f} KB transcript "
          f"(ciphertext) at {info.oid[:10]}")

    # 2. encrypted metadata and the public library entry
    meta = {"title": entry["title"], "series": entry.get("series", ""), "bg": entry.get("bg"),
            "chapters": json.loads(chapters_f.read_text(encoding="utf-8")), "doc": url(doc_path)}
    meta_ct = seal(key, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    public = {
        "id": f"locked-{opaque}", "title": "Locked lecture", "series": "Locked",
        "audio": url(audio_path), "bytes": len(audio_ct), "duration": duration_of(mp3),
        "doc": None, "bg": None, "published": entry.get("published") or dt.date.today().isoformat(),
        "sha256": hashlib.sha256(audio_ct).hexdigest(),
        "locked": {"v": 1,
                   "kdf": {"name": "PBKDF2", "hash": "SHA-256", "iterations": ITERATIONS, "salt": b64(salt)},
                   "meta": {"iv": b64(meta_ct[:12]), "ct": b64(meta_ct[12:])}},
    }
    locked = json.loads(LOCKED_JSON.read_text()) if LOCKED_JSON.exists() else {"lectures": []}
    locked["lectures"] = [l for l in locked["lectures"] if l["id"] != public["id"]] + [public]
    LOCKED_JSON.write_text(json.dumps(locked, indent=2) + "\n")

    # 3. prove the round trip from what was published
    k2 = derive(password, base64.b64decode(public["locked"]["kdf"]["salt"]))
    m = public["locked"]["meta"]
    back = json.loads(unseal(k2, base64.b64decode(m["iv"]) + base64.b64decode(m["ct"])))
    assert back["title"] == entry["title"] and len(back["chapters"]) == len(meta["chapters"])
    assert unseal(k2, audio_ct) == mp3.read_bytes() and unseal(k2, doc_ct) == doc_f.read_bytes()
    try:
        unseal(derive(password + "x", salt), meta_ct)
    except InvalidTag:
        pass                                  # the expected outcome: GCM rejects the wrong key
    else:
        sys.exit("a wrong password decrypted the metadata: this must not happen")
    print(f"locked.json: {public['id']} ({public['duration'] / 60:.1f} min, {len(back['chapters'])} chapters); "
          f"round trip verified, wrong password rejected")


if __name__ == "__main__":
    main()
