"""Stage 0 (HF Job, GPU): transcribe the Teller prompt and held-out clips with Whisper large-v3.

Writes transcripts.json ({"prompts/p00.wav": "...", "heldout/h00.wav": "..."}) back to the
private voice repo. The texts are later snapped to the exact lecture script locally.
"""
import glob
import json
import os

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import pipeline

REPO = "arjun10g/papers-audio-voice"
import sys
DIR = sys.argv[1] if len(sys.argv) > 1 else "train"

d = snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{DIR}/*"])
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
               torch_dtype=torch.float16, device="cuda:0")

out = {}
for f in sorted(glob.glob(f"{d}/{DIR}/*.wav")):
    r = asr(f, return_timestamps=True, generate_kwargs={"language": "english", "task": "transcribe"})
    key = os.path.relpath(f, d)
    out[key] = r["text"].strip()
    print(key, "|", out[key][:100], flush=True)

with open("/tmp/transcripts.json", "w") as fh:
    json.dump(out, fh, indent=1)
HfApi().upload_file(path_or_fileobj="/tmp/transcripts.json", path_in_repo=f"transcripts_{DIR}.json",
                    repo_id=REPO, repo_type="dataset", commit_message="Whisper large-v3 transcripts of Teller clips")
print("DONE", len(out))
