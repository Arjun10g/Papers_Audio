"""Stage 2 (HF Job, GPU): score every bake-off system against real Teller.

Systems are bakeoff/<model>/<prompt>/*.wav plus "teller_real" (the held-out clips themselves).
Per clip:
  sim_ecapa / sim_wavlm  mean cosine similarity to the 16 real held-out Teller clips
                          (excluding the clip itself for teller_real), two independent encoders
  wer                     Whisper large-v3 vs the target text, Whisper's English normaliser
  utmos                   UTMOS22-strong predicted MOS
  rate_ratio              (held-out items only) duration of real Teller / duration of synthesis
                          for the same sentence: 1.0 means Teller's pace
Writes bakeoff/results.json and bakeoff/results.md to the private repo.
"""
import glob
import json
import os
from collections import defaultdict

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

REPO = "arjun10g/papers-audio-voice"
DEV = "cuda:0"
d = snapshot_download(REPO, repo_type="dataset",
                      allow_patterns=["heldout/*", "bakeoff/*", "transcripts*.json", "texts_domain.json"])
tpath = f"{d}/transcripts_exact.json" if os.path.exists(f"{d}/transcripts_exact.json") else f"{d}/transcripts.json"
tr = json.load(open(tpath))
targets = {os.path.basename(k)[:-4]: v for k, v in tr.items() if k.startswith("heldout/")}
targets.update(json.load(open(f"{d}/texts_domain.json")))


def load16(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav, len(wav) / 16000


# systems
systems = {"teller_real": {os.path.basename(f)[:-4]: f for f in sorted(glob.glob(f"{d}/heldout/*.wav"))}}
for f in sorted(glob.glob(f"{d}/bakeoff/*/*/*.wav")):
    model, prompt = f.split("/")[-3], f.split("/")[-2]
    systems.setdefault(f"{model}/{prompt}", {})[os.path.basename(f)[:-4]] = f
print({k: len(v) for k, v in systems.items()}, flush=True)

# models
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402
ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV},
                                       savedir="/tmp/ecapa")
wfe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEV).eval()
utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device=DEV)
norm = asr.tokenizer.normalize


@torch.no_grad()
def embed(wav):
    e1 = ecapa.encode_batch(wav.unsqueeze(0).to(DEV)).squeeze()
    inp = wfe(wav.numpy(), sampling_rate=16000, return_tensors="pt").to(DEV)
    e2 = wavlm(**inp).embeddings.squeeze()
    return torch.nn.functional.normalize(e1, dim=-1).cpu(), torch.nn.functional.normalize(e2, dim=-1).cpu()


real = {k: embed(load16(f)[0]) for k, f in systems["teller_real"].items()}
real_dur = {k: load16(f)[1] for k, f in systems["teller_real"].items()}

rows = []
for sysname, clips in systems.items():
    for item, path in clips.items():
        wav, dur = load16(path)
        e1, e2 = embed(wav)
        refs = [k for k in real if not (sysname == "teller_real" and k == item)]
        sim1 = float(np.mean([float(e1 @ real[k][0]) for k in refs]))
        sim2 = float(np.mean([float(e2 @ real[k][1]) for k in refs]))
        with torch.no_grad():
            mos = float(utmos(wav.unsqueeze(0).to(DEV), 16000).item())
        hyp = asr({"raw": wav.numpy(), "sampling_rate": 16000}, return_timestamps=True,
                  generate_kwargs={"language": "english", "task": "transcribe"})["text"]
        ref_t = targets[item]
        wer = jiwer.wer(norm(ref_t), norm(hyp)) if norm(ref_t).strip() else float("nan")
        row = {"system": sysname, "item": item, "domain": item.startswith("d"), "dur": round(dur, 2),
               "sim_ecapa": round(sim1, 4), "sim_wavlm": round(sim2, 4), "utmos": round(mos, 3),
               "wer": round(wer, 4), "hyp": hyp.strip()}
        if item in real_dur:
            row["rate_ratio"] = round(real_dur[item] / dur, 3)
        rows.append(row)
        print(f"{sysname:22s} {item:4s} sim={sim1:.3f}/{sim2:.3f} mos={mos:.2f} wer={wer:.3f}"
              + (f" rate={row['rate_ratio']:.2f}" if "rate_ratio" in row else ""), flush=True)

# aggregate
meta_rtf = {}
for mf in glob.glob(f"{d}/bakeoff/*/meta.json"):
    m = json.load(open(mf))
    for p, v in m["prompts"].items():
        meta_rtf[f"{m['model']}/{p}"] = {"rtf": v["rtf"], "sr": m["sr"], "vram": m.get("peak_vram_gb")}
agg = defaultdict(dict)
for sysname in systems:
    R = [r for r in rows if r["system"] == sysname]
    H = [r for r in R if not r["domain"]]
    D = [r for r in R if r["domain"]]
    mean = lambda xs, k: round(float(np.nanmean([x[k] for x in xs])), 4) if xs else None
    agg[sysname] = {"n": len(R), "sim_ecapa": mean(R, "sim_ecapa"), "sim_wavlm": mean(R, "sim_wavlm"),
                    "utmos": mean(R, "utmos"), "wer_heldout": mean(H, "wer"), "wer_domain": mean(D, "wer"),
                    "rate_ratio": mean([r for r in H if "rate_ratio" in r], "rate_ratio"),
                    "max_wer": round(max(r["wer"] for r in R), 3), **meta_rtf.get(sysname, {})}

lines = ["| system | SIM ecapa | SIM wavlm | UTMOS | WER held-out | WER domain | max WER | pace vs Teller | RTF | sr |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for s, a in sorted(agg.items(), key=lambda kv: -(kv[1]["sim_ecapa"] or 0)):
    lines.append(f"| {s} | {a['sim_ecapa']} | {a['sim_wavlm']} | {a['utmos']} | {a['wer_heldout']} | "
                 f"{a['wer_domain']} | {a['max_wer']} | {a['rate_ratio']} | {a.get('rtf', '-')} | {a.get('sr', '-')} |")
md = "\n".join(lines)
print(md, flush=True)
os.makedirs("/tmp/res", exist_ok=True)
json.dump({"aggregate": agg, "rows": rows}, open("/tmp/res/results.json", "w"), indent=1)
open("/tmp/res/results.md", "w").write(md + "\n")
HfApi().upload_folder(folder_path="/tmp/res", path_in_repo="bakeoff", repo_id=REPO, repo_type="dataset",
                      commit_message="bake-off scores")
print("DONE", flush=True)
