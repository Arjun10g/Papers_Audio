#!/usr/bin/env python3
"""Encrypt a private lecture for CI, or restore it inside the publishing runner.

Only the encrypted envelope is committed. The password is read from a hidden
prompt locally or LECTURE_PASSWORD in CI. The decrypted lecture and catalog
live under git-ignored private/.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import secrets
import subprocess
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "private"
PENDING = ROOT / "pending_locked"
MAGIC = b"PLS1"
ITERATIONS = 600_000


def password() -> str:
    value = os.environ.get("LECTURE_PASSWORD") or getpass.getpass("Lecture password: ")
    if not value:
        raise SystemExit("empty lecture password")
    return value


def key(value: str, salt: bytes) -> bytes:
    return PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS
    ).derive(value.encode("utf-8"))


def valid_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        raise SystemExit("invalid lecture id")
    return value


def encrypt(lecture_id: str, set_github_secret: bool) -> None:
    lecture_id = valid_id(lecture_id)
    catalog = json.loads((PRIVATE / "catalog.json").read_text(encoding="utf-8"))
    entry = next((item for item in catalog["lectures"] if item["id"] == lecture_id), None)
    if entry is None or not entry.get("locked"):
        raise SystemExit("lecture must have a locked private catalog entry")
    doc = (PRIVATE / "lectures" / f"{lecture_id}.md").read_text(encoding="utf-8")
    opaque = secrets.token_hex(4)
    payload = json.dumps(
        {"id": lecture_id, "entry": entry, "doc": doc, "opaque": opaque},
        ensure_ascii=False,
    ).encode("utf-8")
    salt, iv = os.urandom(16), os.urandom(12)
    pw = password()
    blob = MAGIC + salt + iv + AESGCM(key(pw, salt)).encrypt(iv, payload, MAGIC)
    PENDING.mkdir(exist_ok=True)
    path = PENDING / f"{opaque}.enc"
    path.write_bytes(blob)
    print(f"Created encrypted source: {path.relative_to(ROOT)}")
    if set_github_secret:
        subprocess.run(
            ["gh", "secret", "set", "-R", "Arjun10g/Papers_Audio", "LECTURE_PASSWORD"],
            input=pw,
            text=True,
            check=True,
        )
        print("Set temporary GitHub Actions lecture password secret")


def decrypt_all() -> None:
    paths = sorted(PENDING.glob("*.enc"))
    if not paths:
        raise SystemExit("no encrypted lecture source")
    pw = password()
    entries, ids = [], {}
    (PRIVATE / "lectures").mkdir(parents=True, exist_ok=True)
    for path in paths:
        blob = path.read_bytes()
        if len(blob) < 33 or blob[:4] != MAGIC:
            raise SystemExit("invalid encrypted source")
        salt, iv = blob[4:20], blob[20:32]
        payload = json.loads(AESGCM(key(pw, salt)).decrypt(iv, blob[32:], MAGIC))
        lecture_id = valid_id(payload["id"])
        opaque = payload["opaque"]
        if not re.fullmatch(r"[0-9a-f]{8}", opaque) or path.stem != opaque:
            raise SystemExit("opaque id mismatch")
        if payload["entry"]["id"] != lecture_id or not payload["entry"].get("locked"):
            raise SystemExit("catalog entry mismatch")
        (PRIVATE / "lectures" / f"{lecture_id}.md").write_text(
            payload["doc"], encoding="utf-8"
        )
        entries.append(payload["entry"])
        ids[lecture_id] = opaque
    (PRIVATE / "catalog.json").write_text(
        json.dumps({"lectures": entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (PRIVATE / "locked_ids.json").write_text(json.dumps(ids, indent=2) + "\n")
    (PRIVATE / "ids.txt").write_text("\n".join(ids) + "\n")
    print(f"Restored {len(ids)} private lecture source(s)")


def verify(lecture_id: str) -> None:
    lecture_id = valid_id(lecture_id)
    pw = password()
    matches = 0
    for path in PENDING.glob("*.enc"):
        blob = path.read_bytes()
        if blob[:4] != MAGIC:
            raise SystemExit("invalid encrypted source")
        payload = json.loads(
            AESGCM(key(pw, blob[4:20])).decrypt(blob[20:32], blob[32:], MAGIC)
        )
        if payload["id"] != lecture_id:
            continue
        original = (PRIVATE / "lectures" / f"{lecture_id}.md").read_text(encoding="utf-8")
        catalog = json.loads((PRIVATE / "catalog.json").read_text(encoding="utf-8"))
        entry = next(item for item in catalog["lectures"] if item["id"] == lecture_id)
        assert payload["doc"] == original and payload["entry"] == entry
        assert payload["opaque"] == path.stem
        matches += 1
    if matches != 1:
        raise SystemExit(f"expected one matching envelope, got {matches}")
    print("Encrypted lecture source round trip verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("encrypt", "decrypt-all", "verify"))
    parser.add_argument("lecture_id", nargs="?")
    parser.add_argument("--github-secret", action="store_true")
    args = parser.parse_args()
    if args.action == "encrypt":
        if not args.lecture_id:
            parser.error("encrypt requires lecture_id")
        encrypt(args.lecture_id, args.github_secret)
    elif args.action == "decrypt-all":
        decrypt_all()
    else:
        if not args.lecture_id:
            parser.error("verify requires lecture_id")
        verify(args.lecture_id)


if __name__ == "__main__":
    main()
