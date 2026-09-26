"""Stage 1 (HF Job, GPU): clone Teller with one model and synthesize the bake-off set.

    python stage1_synth.py <voxcpm2|qwen3> <prompt ids, e.g. p01,p06>

For each prompt clip, synthesizes every held-out Teller sentence (ground truth exists) and the
Lecture 1a domain paragraphs, then uploads bakeoff/<model>/<prompt>/<item>.wav + meta.json.
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi, snapshot_download

MODEL, PROMPTS = sys.argv[1], sys.argv[2].split(",")
REPO = "arjun10g/papers-audio-voice"

d = snapshot_download(REPO, repo_type="dataset",
                      allow_patterns=["prompts/*", "heldout/*", "transcripts*.json", "texts_domain.json"])
tpath = f"{d}/transcripts_exact.json" if os.path.exists(f"{d}/transcripts_exact.json") else f"{d}/transcripts.json"
transcripts = json.load(open(tpath))
items = {os.path.basename(k)[:-4]: v for k, v in transcripts.items() if k.startswith("heldout/")}
items.update(json.load(open(f"{d}/texts_domain.json")))
print(f"{MODEL}: {len(items)} texts x {len(PROMPTS)} prompts, transcripts from {os.path.basename(tpath)}", flush=True)

state = {"sr": None}

if MODEL == "voxcpm2":
    from voxcpm import VoxCPM
    m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    state["sr"] = m.tts_model.sample_rate

    def make(prompt_wav, prompt_text):
        def synth(text):
            return np.asarray(m.generate(text=text, prompt_wav_path=prompt_wav, prompt_text=prompt_text,
                                         reference_wav_path=prompt_wav, cfg_value=2.0, inference_timesteps=10),
                              dtype=np.float32).reshape(-1)
        return synth

elif MODEL == "qwen3":
    from qwen_tts import Qwen3TTSModel
    kw = {}
    try:
        import flash_attn  # noqa: F401
        kw["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
                                      dtype=torch.bfloat16, **kw)

    def make(prompt_wav, prompt_text):
        prompt = m.create_voice_clone_prompt(ref_audio=prompt_wav, ref_text=prompt_text, x_vector_only_mode=False)

        def synth(text):
            wavs, sr = m.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt)
            state["sr"] = sr
            return np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return synth
else:
    sys.exit(f"unknown model {MODEL}")

meta = {"model": MODEL, "prompts": {}, "items": {}, "gpu": torch.cuda.get_device_name(0)}
out_root = f"/tmp/out/{MODEL}"
for p in PROMPTS:
    pw, pt = f"{d}/prompts/{p}.wav", transcripts[f"prompts/{p}.wav"]
    synth = make(pw, pt)
    synth("Warming up.")                                  # first call compiles / allocates
    os.makedirs(f"{out_root}/{p}", exist_ok=True)
    tot_audio = tot_wall = 0.0
    for k, text in items.items():
        torch.cuda.synchronize(); t0 = time.time()
        wav = synth(text)
        torch.cuda.synchronize(); wall = time.time() - t0
        secs = len(wav) / state["sr"]
        tot_audio += secs; tot_wall += wall
        sf.write(f"{out_root}/{p}/{k}.wav", wav, state["sr"])
        meta["items"].setdefault(k, text)
        print(f"  {p}/{k}: {secs:5.1f}s audio in {wall:5.1f}s  chars={len(text)}", flush=True)
    meta["prompts"][p] = {"text": pt, "rtf": round(tot_wall / max(tot_audio, 1e-6), 3),
                          "audio_s": round(tot_audio, 1)}
meta["sr"] = state["sr"]
meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
json.dump(meta, open(f"{out_root}/meta.json", "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("sr", "peak_vram_gb", "gpu")}), meta["prompts"], flush=True)
HfApi().upload_folder(folder_path=out_root, path_in_repo=f"bakeoff/{MODEL}", repo_id=REPO, repo_type="dataset",
                      commit_message=f"bake-off samples: {MODEL}")
print("DONE", flush=True)
