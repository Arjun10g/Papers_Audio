"""Stage 4 (HF Job, GPU): LoRA fine-tune VoxCPM2 on Teller, then synthesize the bake-off set.

    python stage4_lora.py <num_iters> <eval steps, e.g. 500,1000> <prompt id, e.g. L01>

Training data: train/*.wav with transcripts_train_exact.json (held-out clips were excluded when
the training clips were cut). Uses VoxCPM's own training script and LoRA config. Uploads the
LoRA checkpoints to loras/teller-voxcpm2/ and samples to bakeoff/voxcpm2lora<step>/<prompt>/.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from huggingface_hub import HfApi, snapshot_download

ITERS, EVAL_STEPS, PROMPT = int(sys.argv[1]), [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
REPO = "arjun10g/papers-audio-voice"
api = HfApi()

d = snapshot_download(REPO, repo_type="dataset",
                      allow_patterns=["train/*", "heldout/*", f"prompts/{PROMPT}.wav", "transcripts_exact.json",
                                      "transcripts_train_exact.json", "texts_domain.json"])
base = snapshot_download("openbmb/VoxCPM2")
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/OpenBMB/VoxCPM.git", "/tmp/VoxCPM"], check=True)

# ── manifest ──
train_tr = json.load(open(f"{d}/transcripts_train_exact.json"))
with open("/tmp/train.jsonl", "w") as fh:
    for k, text in sorted(train_tr.items()):
        fh.write(json.dumps({"audio": f"{d}/{k}", "text": text}) + "\n")
print(f"manifest: {len(train_tr)} clips", flush=True)

# ── config: the official LoRA recipe, pointed at our data ──
cfg = yaml.safe_load(open("/tmp/VoxCPM/conf/voxcpm_v2/voxcpm_finetune_lora.yaml"))
cfg.update({"pretrained_path": base, "train_manifest": "/tmp/train.jsonl", "val_manifest": None,
            "num_iters": ITERS, "max_steps": ITERS, "save_interval": 250, "valid_interval": ITERS + 1,
            "num_workers": 4, "save_path": "/tmp/ckpt", "tensorboard": "/tmp/tb"})
yaml.safe_dump(cfg, open("/tmp/lora.yaml", "w"))
t0 = time.time()
subprocess.run([sys.executable, "scripts/train_voxcpm_finetune.py", "--config_path", "/tmp/lora.yaml"],
               cwd="/tmp/VoxCPM", check=True)
print(f"training took {(time.time() - t0) / 60:.1f} min", flush=True)
print(sorted(os.listdir("/tmp/ckpt")), flush=True)
api.upload_folder(folder_path="/tmp/ckpt", path_in_repo="loras/teller-voxcpm2", repo_id=REPO, repo_type="dataset",
                  allow_patterns=["step_*/lora_weights.safetensors", "step_*/lora_config.json"],
                  commit_message=f"Teller LoRA for VoxCPM2 ({ITERS} iters)")

# ── synthesize the bake-off set with each checkpoint ──
from voxcpm.core import VoxCPM  # noqa: E402
from voxcpm.model.voxcpm import LoRAConfig  # noqa: E402

tr = json.load(open(f"{d}/transcripts_exact.json"))
items = {os.path.basename(k)[:-4]: v for k, v in tr.items() if k.startswith("heldout/")}
items.update(json.load(open(f"{d}/texts_domain.json")))
pw, pt = f"{d}/prompts/{PROMPT}.wav", tr[f"prompts/{PROMPT}.wav"]
for step in EVAL_STEPS:
    ck = f"/tmp/ckpt/step_{step:07d}"
    info = json.load(open(f"{ck}/lora_config.json"))
    m = VoxCPM.from_pretrained(hf_model_id=base, load_denoiser=False, optimize=True,
                               lora_config=LoRAConfig(**info["lora_config"]), lora_weights_path=ck)
    sr = m.tts_model.sample_rate
    out = f"/tmp/out/voxcpm2lora{step}/{PROMPT}"
    os.makedirs(out, exist_ok=True)
    m.generate(text="Warming up.", prompt_wav_path=pw, prompt_text=pt)
    ta = tw = 0.0
    for k, text in items.items():
        t1 = time.time()
        wav = np.asarray(m.generate(text=text, prompt_wav_path=pw, prompt_text=pt, reference_wav_path=pw,
                                    cfg_value=2.0, inference_timesteps=10), dtype=np.float32).reshape(-1)
        tw += time.time() - t1; ta += len(wav) / sr
        sf.write(f"{out}/{k}.wav", wav, sr)
        print(f"  step {step} {k}: {len(wav) / sr:5.1f}s", flush=True)
    meta = {"model": f"voxcpm2lora{step}", "sr": sr, "gpu": torch.cuda.get_device_name(0),
            "prompts": {PROMPT: {"text": pt, "rtf": round(tw / ta, 3), "audio_s": round(ta, 1)}}}
    json.dump(meta, open(f"/tmp/out/voxcpm2lora{step}/meta.json", "w"), indent=1)
    api.upload_folder(folder_path=f"/tmp/out/voxcpm2lora{step}", path_in_repo=f"bakeoff/voxcpm2lora{step}",
                      repo_id=REPO, repo_type="dataset", commit_message=f"bake-off samples: LoRA step {step}")
    del m
    torch.cuda.empty_cache()
print("DONE", flush=True)
