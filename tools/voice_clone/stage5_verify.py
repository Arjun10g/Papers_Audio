"""Stage 5 (HF Job, GPU): verify full cloned-voice renders against their scripts.

    python stage5_verify.py <voice> <lecture id>...

Transcribes renders/<voice>/<id>.mp3 with Whisper large-v3 (timestamped), splits the transcript
at the chapter markers, and reports word error per chapter against the narration script, so a
skipped, garbled or hallucinated passage anywhere in a lecture shows up. Writes
renders/<voice>/<id>.verify.json.
"""
import json
import sys

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import HfApi, snapshot_download
from transformers import pipeline

VOICE, IDS = sys.argv[1], sys.argv[2:]
REPO = "arjun10g/papers-audio-voice"
d = snapshot_download(REPO, repo_type="dataset",
                      allow_patterns=[f"renders/{VOICE}/*", "render_bundle/*", "render_bundle/**/*"])
sys.path.insert(0, f"{d}/render_bundle/tools")
import generate_lecture as gl  # noqa: E402

# Sequential long-form decoding (no chunk_length_s): slower than batched chunking, but it has no
# overlapping chunk seams, which is where Whisper duplicates text and hallucinates stock phrases.
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16,
               device="cuda:0")
norm = asr.tokenizer.normalize

for lid in IDS:
    blocks = gl.markdown_to_blocks(open(f"{d}/render_bundle/lectures/{lid}.md").read())
    chapters = json.load(open(f"{d}/renders/{VOICE}/{lid}.chapters.json"))
    # script text per chapter, in the same order the narrator spoke it
    titles = [c["title"] for c in chapters]
    script = {t: [] for t in titles}
    cur = titles[0] if titles[0] == "Introduction" else None
    for b in blocks:
        if b.level == 2 and b.text.rstrip(".") in script:
            cur = b.text.rstrip(".")
        if cur:
            script[cur].append(b.text)
    wav, sr = sf.read(f"{d}/renders/{VOICE}/{lid}.mp3", dtype="float32", always_2d=True)
    wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16000).numpy()
    out = asr({"raw": wav, "sampling_rate": 16000}, return_timestamps=True,
              generate_kwargs={"language": "english", "task": "transcribe", "condition_on_prev_tokens": False,
                               "temperature": (0.0, 0.2, 0.4), "compression_ratio_threshold": 1.35,
                               "logprob_threshold": -1.0, "no_speech_threshold": 0.6})
    starts = [c["start"] for c in chapters] + [1e9]
    heard = {t: [] for t in titles}
    for ch in out["chunks"]:
        t0 = ch["timestamp"][0] or 0.0
        i = max(k for k in range(len(titles)) if starts[k] <= t0 + 0.5)
        heard[titles[i]].append(ch["text"])
    rows = []
    for t in titles:
        ref, hyp = norm(" ".join(script[t])), norm(" ".join(heard[t]))
        w = jiwer.wer(ref, hyp) if ref.strip() else float("nan")
        rows.append({"chapter": t, "wer": round(w, 4), "ref_words": len(ref.split()), "hyp_words": len(hyp.split())})
        print(f"{lid:24s} {w:6.3f}  ref={len(ref.split()):5d} hyp={len(hyp.split()):5d}  {t}", flush=True)
    allref = norm(" ".join(" ".join(v) for v in script.values()))
    allhyp = norm(" ".join(c["text"] for c in out["chunks"]))
    total = jiwer.wer(allref, allhyp)
    print(f"{lid}: overall WER {total:.4f}", flush=True)
    json.dump({"id": lid, "overall_wer": round(total, 4), "chapters": rows, "transcript": out["text"]},
              open(f"/tmp/{lid}.verify.json", "w"), indent=1)
    HfApi().upload_file(path_or_fileobj=f"/tmp/{lid}.verify.json", path_in_repo=f"renders/{VOICE}/{lid}.verify.json",
                        repo_id=REPO, repo_type="dataset", commit_message=f"verify {lid}")
print("DONE", flush=True)
