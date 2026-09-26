"""Stage 3 (HF Job, GPU): narrate a full lecture with a cloned voice.

    python stage3_render.py <lecture id> <voice name>

Pulls render_bundle/ (tools/generate_lecture.py, catalog.json, lectures/*.md) from the private
voice repo, runs the clone engine, and uploads renders/<id>.mp3 + renders/<id>.chapters.json.
"""
import os
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, snapshot_download

LID, VOICE = sys.argv[1], sys.argv[2]
REPO = "arjun10g/papers-audio-voice"

d = snapshot_download(REPO, repo_type="dataset", allow_patterns=["render_bundle/*", "render_bundle/**/*"])
root = "/tmp/bundle"
shutil.copytree(f"{d}/render_bundle", root, dirs_exist_ok=True)
subprocess.run([sys.executable, "tools/generate_lecture.py", LID, "--engine", "clone", "--voice", VOICE, "--force"],
               cwd=root, check=True)
api = HfApi()
for name in (f"{LID}.mp3", f"{LID}.chapters.json"):
    api.upload_file(path_or_fileobj=f"{root}/audio/{name}", path_in_repo=f"renders/{VOICE}/{name}",
                    repo_id=REPO, repo_type="dataset", commit_message=f"render {LID} in {VOICE}")
print("DONE", LID, flush=True)
