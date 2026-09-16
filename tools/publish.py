#!/usr/bin/env python3
"""Publish the library to the Hugging Face dataset the phone app streams from.

    python tools/publish.py                # sync audio/ + lectures/ + library.json
    python tools/publish.py --dry-run      # show what would change
    python tools/publish.py --list-missing # slugs with a transcript but no audio anywhere

What it does, in order:

1. Reads catalog.json — the ordered list of lectures and the only file you edit
   by hand.
2. Compares audio/<id>.mp3 (if present locally) against what is already on the
   dataset by sha256 and uploads only new or changed files, together with every
   lectures/<id>.md, in one commit.
3. Builds library.json: duration, size, sha, transcript URL, chapter markers.
   Audio URLs are pinned to the last commit that changed audio (the app keys
   offline downloads by URL, so they must not move otherwise); transcripts
   point at main so edits show up immediately.
   Durations/chapters for lectures whose audio is not on this machine (the
   normal case in CI) are carried over from the previous library.json.
4. Uploads library.json (second commit) and writes a copy to app/library.json
   so the deployed app has an offline fallback.

Auth: --token, else $HF_TOKEN, else the token saved by `hf auth login`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.json"
AUDIO = ROOT / "audio"
LECTURES = ROOT / "lectures"
APP_LIBRARY = ROOT / "app" / "library.json"

REPO_ID = "arjun10g/papers-audio"
REPO_TYPE = "dataset"
HUB = "https://huggingface.co"

README = """---
pretty_name: Papers, as Audio
tags:
  - audio
  - lectures
  - text-to-speech
---

# Papers, as Audio

Narrated long-form lectures on statistics and machine learning, streamed by the
[Papers, as Audio](https://arjun10g.github.io/Papers_Audio/) phone app.

- `library.json` — the catalog the app reads (titles, durations, chapter markers, file paths)
- `audio/<id>.mp3` — one file per lecture (Kokoro-82M `af_heart`, 96 kbps mono)
- `lectures/<id>.md` — transcripts

Published automatically from https://github.com/Arjun10g/Papers_Audio.
"""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def duration_of(path: Path) -> float:
    try:
        from mutagen.mp3 import MP3
        return round(float(MP3(str(path)).info.length), 2)
    except Exception:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True,
        ).stdout.strip()
        return round(float(out), 2)


def load_catalog() -> dict:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    seen = set()
    for lec in cat["lectures"]:
        if lec["id"] in seen:
            sys.exit(f"catalog.json: duplicate id {lec['id']!r}")
        seen.add(lec["id"])
    return cat


def remote_files(api, repo_exists: bool) -> dict[str, dict]:
    """path -> {size, sha256|None} for everything currently on the dataset."""
    if not repo_exists:
        return {}
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)
    out = {}
    for s in info.siblings or []:
        lfs = getattr(s, "lfs", None)
        sha = None
        if lfs:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        out[s.rfilename] = {"size": getattr(s, "size", None), "sha256": sha}
    return out


def previous_library(api, repo_exists: bool) -> tuple[dict[str, dict], str | None]:
    """(id -> lecture entry, pinned audio revision) from the published library.json."""
    if not repo_exists:
        return {}, None
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO_ID, "library.json", repo_type=REPO_TYPE, force_download=True)
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        return {lec["id"]: lec for lec in data.get("lectures", [])}, data.get("revision")
    except Exception as e:  # first publish, or a corrupt file — rebuild from scratch
        print(f"  (no previous library.json: {type(e).__name__})")
        return {}, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN") or None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-missing", action="store_true",
                    help="print ids that have lectures/<id>.md but no audio locally or remotely")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=args.token)
    cat = load_catalog()

    try:
        api.repo_info(REPO_ID, repo_type=REPO_TYPE)
        repo_exists = True
    except RepositoryNotFoundError:
        repo_exists = False

    remote = remote_files(api, repo_exists)
    prev, prev_rev = previous_library(api, repo_exists)

    # ── Decide what to upload ──
    ops: list = []
    local_sha: dict[str, str] = {}
    for lec in cat["lectures"]:
        slug = lec["id"]
        mp3 = AUDIO / f"{slug}.mp3"
        if mp3.exists():
            sha = sha256_of(mp3)
            local_sha[slug] = sha
            if remote.get(f"audio/{slug}.mp3", {}).get("sha256") != sha:
                ops.append(CommitOperationAdd(path_in_repo=f"audio/{slug}.mp3", path_or_fileobj=str(mp3)))
        doc = LECTURES / f"{slug}.md"
        if doc.exists():
            ops.append(CommitOperationAdd(path_in_repo=f"lectures/{slug}.md", path_or_fileobj=str(doc)))

    if args.list_missing:
        for lec in cat["lectures"]:
            slug = lec["id"]
            has_audio = (AUDIO / f"{slug}.mp3").exists() or f"audio/{slug}.mp3" in remote
            if (LECTURES / f"{slug}.md").exists() and not has_audio:
                print(slug)
        return

    if not repo_exists or "README.md" not in remote:
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=README.encode("utf-8")))

    audio_ops = [o.path_in_repo for o in ops if o.path_in_repo.startswith("audio/")]
    print(f"{len(audio_ops)} audio file(s) to upload: {', '.join(audio_ops) or '-'}")
    print(f"{sum(1 for o in ops if o.path_in_repo.startswith('lectures/'))} transcript(s) to sync")

    if args.dry_run:
        print("dry run — nothing uploaded")
        revision = "main"
    else:
        if not repo_exists:
            api.create_repo(REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)
            print(f"created {HUB}/datasets/{REPO_ID}")
        if ops:
            info = api.create_commit(
                REPO_ID, repo_type=REPO_TYPE, operations=ops,
                commit_message=f"Sync {len(audio_ops)} audio file(s) and transcripts",
            )
            print(f"committed {info.oid[:10]}")
        else:
            print("nothing to upload")
        remote = remote_files(api, True)

        # The app keys its offline downloads by full URL, so `base` must only
        # move when audio actually changes — not on every library.json commit.
        if audio_ops or not prev_rev:
            revision = api.repo_info(REPO_ID, repo_type=REPO_TYPE).sha
        else:
            revision = prev_rev

    # ── Build library.json ──
    base = f"{HUB}/datasets/{REPO_ID}/resolve/{revision}/"
    doc_base = f"{HUB}/datasets/{REPO_ID}/resolve/main/"     # transcripts: always the latest
    lectures_out = []
    for n, lec in enumerate(cat["lectures"], 1):
        slug = lec["id"]
        key = f"audio/{slug}.mp3"
        mp3 = AUDIO / f"{slug}.mp3"
        r = remote.get(key)
        if r is None and not mp3.exists():
            print(f"  skip {slug}: no audio locally or on the dataset")
            continue
        sha = local_sha.get(slug) or (r or {}).get("sha256")
        size = mp3.stat().st_size if mp3.exists() else (r or {}).get("size")
        old = prev.get(slug)
        same_as_before = bool(old and sha and old.get("sha256") == sha)

        chapters_file = AUDIO / f"{slug}.chapters.json"
        if mp3.exists():
            duration = duration_of(mp3)
            chapters = (json.loads(chapters_file.read_text(encoding="utf-8"))
                        if chapters_file.exists() else (old.get("chapters") if same_as_before and old else None))
        elif same_as_before:
            duration, chapters = old.get("duration"), old.get("chapters")
        else:
            duration, chapters = None, None
            print(f"  warning: {slug} duration unknown (audio not local, no previous entry)")

        entry = {
            "id": slug,
            "n": n,
            "title": lec["title"],
            "series": lec.get("series", ""),
            "audio": key,
            "bytes": size,
            "duration": duration,
            "doc": f"{doc_base}lectures/{slug}.md" if (LECTURES / f"{slug}.md").exists() else None,
            "bg": lec.get("bg"),
            "published": lec.get("published"),
            "sha256": sha,
        }
        if chapters:
            entry["chapters"] = chapters
        lectures_out.append(entry)

    library = {
        "schema": 1,
        "title": cat.get("title", "Papers, as Audio"),
        "generated": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "repo": REPO_ID,
        "revision": revision,
        "base": base,
        "lectures": lectures_out,
    }
    blob = json.dumps(library, indent=2, ensure_ascii=False) + "\n"
    APP_LIBRARY.parent.mkdir(exist_ok=True)
    APP_LIBRARY.write_text(blob, encoding="utf-8")
    print(f"library.json: {len(lectures_out)} lectures → {APP_LIBRARY.relative_to(ROOT)}")

    if args.dry_run:
        return
    api.upload_file(
        path_or_fileobj=io.BytesIO(blob.encode("utf-8")), path_in_repo="library.json",
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        commit_message=f"library.json: {len(lectures_out)} lectures @ {revision[:10]}",
    )
    print(f"published {HUB}/datasets/{REPO_ID}/resolve/main/library.json")


if __name__ == "__main__":
    main()
