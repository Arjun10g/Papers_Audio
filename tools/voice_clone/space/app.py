"""Teller TTS: the Papers, as Audio narrator voice on a private ZeroGPU Space.

VoxCPM2 plus the Teller LoRA, loaded once at startup. The voice profile (prompt clip,
transcript, LoRA weights) is read from the private dataset arjun10g/papers-audio-voice with
the Space's HF_TOKEN secret.

API used by tools/generate_lecture.py (engine "clone" with a `space` in the voice profile):
    /synth(texts_json, voice, seed) -> .npz with sr, gpu_s and int16 clips c0..cN
Many sentences go in one call, because every call pays ZeroGPU's allocation overhead.
"""
import json
import os
import tempfile
import time

import spaces  # noqa: F401  (must be imported before torch on ZeroGPU)
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

import voxcpm.model.voxcpm2 as _voxcpm2

# VoxCPM loads LoRA weights straight onto the GPU through safetensors, which bypasses ZeroGPU's
# startup CUDA emulation ("No CUDA GPUs are available"). Read onto the CPU, then move: `.to()`
# is what the emulation supports.
_load_file = _voxcpm2.load_file


def _load_file_via_cpu(path, device="cpu"):
    state = _load_file(path, device="cpu")
    return state if str(device) == "cpu" else {k: v.to(device) for k, v in state.items()}


_voxcpm2.load_file = _load_file_via_cpu

VOICE_REPO = "arjun10g/papers-audio-voice"
root = snapshot_download(VOICE_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                         allow_patterns=["voices/*", "voices/**/*"])

VOICES, MODELS = {}, {}
for name in sorted(os.listdir(f"{root}/voices")):
    path = f"{root}/voices/{name}"
    cfg = json.load(open(f"{path}/voice.json"))
    if cfg.get("model") != "voxcpm2":
        continue
    cfg["prompt_wav"] = f"{path}/{cfg['prompt_wav']}"
    kw = {}
    if cfg.get("lora"):
        info = json.load(open(f"{path}/{cfg['lora']}/lora_config.json"))
        kw = {"lora_config": LoRAConfig(**info["lora_config"]), "lora_weights_path": f"{path}/{cfg['lora']}"}
    # optimize=False: ZeroGPU does not support torch.compile
    MODELS[name] = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, **kw)
    VOICES[name] = cfg
print("voices:", list(VOICES), flush=True)


def _gpu_seconds(texts_json, voice="teller", seed=0):
    """Time budget for one call: generous for the text sent, capped so a call stays schedulable."""
    chars = sum(len(t) for t in json.loads(texts_json))
    return int(min(240, 25 + chars / 19.3 * 0.9))


@spaces.GPU(duration=_gpu_seconds)
def _synth(texts_json, voice, seed):
    cfg, model = VOICES[voice], MODELS[voice]
    p = cfg.get("params", {})
    t0 = time.time()
    clips = []
    for i, text in enumerate(json.loads(texts_json)):
        torch.manual_seed(int(seed) + i)
        wav = model.generate(text=text, prompt_wav_path=cfg["prompt_wav"], prompt_text=cfg["prompt_text"],
                             reference_wav_path=cfg["prompt_wav"], cfg_value=p.get("cfg_value", 2.0),
                             inference_timesteps=p.get("inference_timesteps", 10))
        wav = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
        clips.append((wav * 32767).astype(np.int16))
    return clips, time.time() - t0


def synth(texts_json: str, voice: str = "teller", seed: float = 0):
    if voice not in VOICES:
        raise gr.Error(f"unknown voice {voice!r}; have {list(VOICES)}")
    clips, gpu_s = _synth(texts_json, voice, seed)
    out = tempfile.NamedTemporaryFile(suffix=".npz", delete=False).name
    np.savez_compressed(out, sr=MODELS[voice].tts_model.sample_rate, gpu_s=gpu_s,
                        **{f"c{i}": c for i, c in enumerate(clips)})
    return out


def preview(text: str, voice: str = "teller"):
    clips, _ = _synth(json.dumps([text]), voice, 0)
    return MODELS[voice].tts_model.sample_rate, clips[0]


with gr.Blocks(title="Teller TTS") as demo:
    gr.Markdown("### Teller TTS\nPrivate narrator voice for *Papers, as Audio*.")
    with gr.Tab("Preview"):
        t = gr.Textbox(label="Text", value="Welcome back. Today we are going to talk about memory bandwidth.")
        v = gr.Dropdown(choices=list(VOICES), value="teller", label="Voice")
        gr.Button("Speak").click(preview, [t, v], gr.Audio(label="Audio"), api_name="preview")
    with gr.Tab("Batch API"):
        tj = gr.Textbox(label="texts_json (JSON list of strings)")
        vv = gr.Textbox(value="teller", label="voice")
        sd = gr.Number(value=0, label="seed")
        gr.Button("Synthesize").click(synth, [tj, vv, sd], gr.File(label="clips.npz"), api_name="synth")

demo.queue(default_concurrency_limit=1).launch()
