#!/usr/bin/env python3
"""Narrate a lecture with Kokoro (free, local, no API key) and record chapters.

    python tools/generate_lecture.py <id> [--voice af_heart] [--speed 1.0] [--force]

Reads   lectures/<id>.md
Writes  audio/<id>.mp3            96 kbps mono, peak-normalised
        audio/<id>.chapters.json  [{"title": ..., "start": seconds}] — one entry
                                  per "## " heading, so the app can jump between
                                  sections of a single lecture.

The markdown is turned into narration rather than read verbatim: headings are
spoken with a longer pause around them, emphasis/links/code spans lose their
markup, fenced code blocks and horizontal rules are dropped, list items and
table rows become sentences. Same Kokoro-82M weights and `af_heart` voice as
every other lecture, so the whole library sounds like one narrator.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 24_000      # an engine may raise this (the clone engine uses its model's rate)
BITRATE = "96k"
GAP_PARA = 0.35       # seconds of silence after a paragraph
GAP_HEADING = 0.9     # before a heading
GAP_AFTER_HEADING = 0.45
PEAK = 0.891          # Kokoro renders quiet; normalise the whole file to -1 dBFS


@dataclass
class Block:
    text: str
    level: int = 0        # 0 = paragraph, 1/2/3… = heading level


# ── Markdown → narration ──

def clean_inline(s: str) -> str:
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)                 # images
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)              # links → text
    s = re.sub(r"\[\^[^\]]+\]", "", s)                           # footnote refs
    s = re.sub(r"<[^>]+>", "", s)                                # html tags
    s = re.sub(r"`([^`]*)`", r"\1", s)                           # code spans
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", s)
    s = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"\1", s)
    s = re.sub(r"\\\((.+?)\\\)", r"\1", s)                       # \( math \)
    s = re.sub(r"\$\$(.+?)\$\$", r"\1", s)
    s = re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", r"\1", s)
    s = html.unescape(s)
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", " - ").replace("–", "-").replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def sentence(s: str) -> str:
    """Make sure a fragment (list item, table row, heading) ends like a sentence."""
    s = s.rstrip()
    return s if not s or s[-1] in ".!?:;" else s + "."


CODE_LANGS = {"python", "py", "bash", "sh", "shell", "zsh", "json", "yaml", "yml", "toml", "ini",
              "c", "cpp", "cuda", "rust", "go", "js", "javascript", "ts", "typescript", "sql",
              "diff", "console", "mermaid", "dockerfile", "makefile", "html", "css", "xml"}
CODE_SIGNS = re.compile(r"\b(def|import|return|flowchart|graph|sequenceDiagram|for|while|if|else)\b|[{};]|==|\$ |-->")


def equation_to_speech(s: str) -> str:
    """Read a plain-text equation the way a lecturer would say it."""
    s = s.strip().rstrip(".,;")
    s = re.sub(r"\b(sum|mean|max|min|prod)_\{?(\w+)\}?\s*\(", r"the \1 over \2 of (", s)
    s = re.sub(r"\bsolve\(", "solve of (", s)
    s = re.sub(r"shape\s*\[([^\]]+)\]", lambda m: "shape " + " by ".join(x.strip() for x in m.group(1).split(",")), s)
    s = re.sub(r"\^T\b", " transpose", s)
    s = re.sub(r"\^\{([^}]*)\}", r" to the \1", s)
    s = re.sub(r"\^(\w+)", r" to the \1", s)
    s = re.sub(r"_\{([^}]*)\}", lambda m: " " + m.group(1).replace(",", " "), s)
    s = re.sub(r"_(\w+)", r" \1", s)
    s = re.sub(r"(\w)-(\w)", r"\1 minus \2", s)
    s = s.replace("=", " equals ").replace("+", " plus ").replace("*", " times ").replace("/", " over ")
    s = re.sub(r"(?<=\S)\s+-\s+(?=\S)", " minus ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def code_to_speech(lang: str, lines: list[str]) -> list[str]:
    """Short fenced blocks are usually equations in these lectures: speak them.
    Real code and diagrams are skipped — nobody wants them read aloud."""
    lines = [l.rstrip() for l in lines if l.strip()]
    if not lines or len(lines) > 6 or lang in CODE_LANGS or CODE_SIGNS.search(" ".join(lines)):
        return []
    out = []
    for l in lines:
        for frag in re.split(r"\s{3,}", l.strip()):      # aligned "z = W h,      z in R^E"
            spoken = equation_to_speech(frag)
            if spoken:
                out.append(sentence(spoken))
    return out


def markdown_to_blocks(md: str) -> list[Block]:
    blocks: list[Block] = []
    para: list[str] = []
    in_code = False
    code_lang = ""
    code_lines: list[str] = []
    skip_level = 0          # >0 while inside a "Contents" section: audio has chapter markers instead

    def flush() -> None:
        nonlocal para
        if para:
            text = clean_inline(" ".join(para))
            if text:
                blocks.append(Block(text))
        para = []

    for raw in md.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush()
            if in_code:
                if not skip_level:
                    blocks.extend(Block(s) for s in code_to_speech(code_lang, code_lines))
                code_lines = []
            else:
                code_lang = stripped.strip("`~").strip().lower()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush()
            continue
        if re.fullmatch(r"\s*([-*_]\s*){3,}", line):            # horizontal rule
            flush()
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            flush()
            level, text = len(m.group(1)), clean_inline(m.group(2))
            if skip_level and level <= skip_level:
                skip_level = 0
            if re.fullmatch(r"(table of )?contents|toc|outline", text.strip(" .:").lower()):
                skip_level = level
                continue
            if text and not skip_level:
                blocks.append(Block(text, level=level))
            continue
        if skip_level:
            continue
        if re.match(r"^\s*\|?\s*:?-{2,}", line) and "|" in line:  # table separator row
            continue
        if line.lstrip().startswith("|"):                       # table row → sentence
            cells = [clean_inline(c) for c in line.strip().strip("|").split("|")]
            cells = [c for c in cells if c]
            if cells:
                flush()
                blocks.append(Block(sentence(", ".join(cells))))
            continue
        m = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$", line)     # list item → sentence
        if m:
            flush()
            text = clean_inline(m.group(1))
            if text:
                blocks.append(Block(sentence(text)))
            continue
        line = re.sub(r"^\s*>\s?", "", line)                       # blockquote marker
        para.append(line.strip())
    flush()
    return blocks


# ── Synthesis ──

def configure_espeak() -> None:
    """On macOS the espeakng-loader wheel's dylib looks for its data at a path
    baked in on the wheel's build machine and then exits the whole process.
    Point misaki at the Homebrew install when there is one (brew install
    espeak-ng). Linux CI runners are fine with the bundled loader."""
    lib, data = Path("/opt/homebrew/lib/libespeak-ng.dylib"), Path("/opt/homebrew/share/espeak-ng-data")
    if sys.platform == "darwin" and lib.exists() and data.exists():
        import espeakng_loader
        espeakng_loader.get_library_path = lambda: str(lib)
        espeakng_loader.get_data_path = lambda: str(data)
        os.environ.setdefault("ESPEAK_DATA_PATH", str(data))
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(lib))


def synthesize_kokoro(blocks: list[Block], voice: str, speed: float):
    """Yield (block, audio float32) in order; silence gaps are added by the caller."""
    configure_espeak()
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    for b in blocks:
        chunks = []
        for _graphemes, _phonemes, audio in pipe(b.text, voice=voice, speed=speed, split_pattern=r"\n+"):
            if audio is not None:
                chunks.append(np.asarray(audio, dtype=np.float32))
        yield b, (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32))


SPEECHIFY_URL = "https://api.sws.speechify.com/v1/audio/speech"
SPEECHIFY_MAX = 1900          # characters per request (API limit is 2000 for plain text)


def speechify_request(text: str, voice: str, key: str) -> bytes:
    """One TTS call → mp3 bytes, retrying on rate limits and server errors."""
    import base64
    import urllib.error
    import urllib.request
    body = json.dumps({"input": text, "voice_id": voice, "audio_format": "mp3",
                       "language": "en-US"}).encode("utf-8")
    req = urllib.request.Request(SPEECHIFY_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return base64.b64decode(json.load(r)["audio_data"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f"Speechify {e.code}: {e.read()[:300]!r}")
    raise SystemExit("Speechify: gave up after retries")


def decode_mp3(data: bytes) -> np.ndarray:
    """mp3 bytes → float32 mono at SAMPLE_RATE, so both engines share one encoder."""
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        input=data, capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype="<f4").astype(np.float32)


def synthesize_speechify(blocks: list[Block], voice: str, speed: float):
    """Same contract as synthesize_kokoro, via the Speechify API.

    Consecutive paragraphs are batched into one request (cheaper, and the
    voice keeps its flow across them); a heading always starts a new request
    so chapter timestamps stay exact. Blocks after the first in a batch are
    yielded with empty audio, which keeps the caller's bookkeeping simple."""
    key = os.environ.get("SPEECHIFY_API_KEY")
    if not key:
        sys.exit("set SPEECHIFY_API_KEY (never hardcode it; the old scripts leaked one)")
    if speed != 1.0:
        print("  note: --speed is ignored for the speechify engine")

    def split_long(text: str) -> list[str]:
        parts, cur = [], ""
        for s in re.split(r"(?<=[.?!])\s+", text):
            if len(cur) + len(s) + 1 > SPEECHIFY_MAX and cur:
                parts.append(cur.strip()); cur = ""
            cur += s + " "
        return parts + ([cur.strip()] if cur.strip() else [])

    batch: list[Block] = []

    def flush():
        if not batch:
            return
        text = "\n\n".join(b.text for b in batch)
        audio = np.concatenate([decode_mp3(speechify_request(t, voice, key)) for t in split_long(text)])
        yield batch[0], audio
        for b in batch[1:]:
            yield b, np.zeros(0, dtype=np.float32)
        batch.clear()

    for b in blocks:
        if b.level or sum(len(x.text) + 2 for x in batch) + len(b.text) > SPEECHIFY_MAX:
            yield from flush()
        batch.append(b)
        if b.level:                      # a heading is spoken on its own
            yield from flush()
    yield from flush()


VOICE_REPO = "arjun10g/papers-audio-voice"     # private: prompt clips for cloned voices


def load_voice(name: str) -> dict:
    """voices/<name>/voice.json + its prompt clip, from ./voices (git-ignored) or the
    private voice repo on Hugging Face (needs HF_TOKEN with read access)."""
    local = ROOT / "voices" / name
    if not (local / "voice.json").exists():
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(VOICE_REPO, repo_type="dataset",
                                       allow_patterns=[f"voices/{name}/*", f"voices/{name}/**/*"])) / "voices" / name
    if not (local / "voice.json").exists():
        sys.exit(f"no voice profile voices/{name}/voice.json (locally or in {VOICE_REPO})")
    cfg = json.loads((local / "voice.json").read_text(encoding="utf-8"))
    cfg["prompt_wav"] = str(local / cfg["prompt_wav"])
    if cfg.get("lora"):
        cfg["lora"] = str(local / cfg["lora"])
    return cfg


def clone_backend(cfg: dict):
    """(synth(text) -> float32 mono, sample_rate) for an open-weight cloning model.
    Model packages are imported lazily: `pip install voxcpm` or `pip install qwen-tts`."""
    model, p = cfg["model"], cfg.get("params", {})
    if model == "voxcpm2":
        from voxcpm import VoxCPM
        kw = {}
        if cfg.get("lora"):              # a LoRA fine-tuned on the voice, shipped inside the profile
            from voxcpm.model.voxcpm import LoRAConfig
            info = json.loads((Path(cfg["lora"]) / "lora_config.json").read_text(encoding="utf-8"))
            kw = {"lora_config": LoRAConfig(**info["lora_config"]), "lora_weights_path": cfg["lora"]}
        m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, **kw)

        def synth(text: str) -> np.ndarray:
            return np.asarray(m.generate(text=text, prompt_wav_path=cfg["prompt_wav"],
                                         prompt_text=cfg["prompt_text"], reference_wav_path=cfg["prompt_wav"],
                                         cfg_value=p.get("cfg_value", 2.0),
                                         inference_timesteps=p.get("inference_timesteps", 10)),
                              dtype=np.float32).reshape(-1)
        return synth, m.tts_model.sample_rate
    if model == "qwen3":
        import torch
        from qwen_tts import Qwen3TTSModel
        m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
                                          dtype=torch.bfloat16)
        prompt = m.create_voice_clone_prompt(ref_audio=cfg["prompt_wav"], ref_text=cfg["prompt_text"],
                                             x_vector_only_mode=False)
        sr_box = {}

        def synth(text: str) -> np.ndarray:
            wavs, sr = m.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt)
            sr_box["sr"] = sr
            return np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        synth("Warming up.")
        return synth, sr_box["sr"]
    sys.exit(f"unknown clone model {model!r}")


def trim_silence(a: np.ndarray, sr: int, keep: float = 0.05) -> np.ndarray:
    """Cut leading/trailing near-silence so the caller's pauses are the only pauses."""
    if not a.size:
        return a
    win = max(1, int(sr * 0.01))
    env = np.sqrt(np.convolve(a * a, np.ones(win) / win, mode="same"))
    loud = np.flatnonzero(env > max(1e-4, 0.02 * env.max()))
    if not loud.size:
        return a[:0]
    pad = int(sr * keep)
    return a[max(0, loud[0] - pad): loud[-1] + pad]


class QuotaPause(Exception):
    """The free ZeroGPU allowance for today is used up; finished chunks are cached, rerun later."""


class SpaceBackend:
    """Batched calls to the private ZeroGPU Space named in the voice profile.

    ZeroGPU is free within a daily allowance (40 GPU-minutes for PRO). Past it, Hugging Face
    bills any prepaid credits, so a local ledger stops at ZEROGPU_BUDGET_MIN (default 36) per
    rolling 24 hours, and a quota error from the Space also pauses the render."""

    def __init__(self, cfg: dict, voice: str, cache_dir: Path):
        from gradio_client import Client
        self.voice, self.cfg = voice, cfg
        self.client = Client(cfg["space"], token=os.environ.get("HF_TOKEN"), verbose=False)
        self.ledger = cache_dir.parent / "zerogpu_ledger.json"
        self.budget = float(os.environ.get("ZEROGPU_BUDGET_MIN", "36")) * 60
        self.sr = None

    def _used(self) -> float:
        try:
            rows = json.loads(self.ledger.read_text())
        except (FileNotFoundError, ValueError):
            rows = []
        return sum(s for t, s in rows if time.time() - t < 86_400), rows

    def synth_many(self, texts: list[str], seed: int) -> list[np.ndarray]:
        used, rows = self._used()
        est = sum(len(t) for t in texts) / self.cfg.get("chars_per_sec", 19.0) * 0.9 + 10
        if used + est > self.budget:
            raise QuotaPause(f"{used / 60:.1f} of {self.budget / 60:.0f} free GPU-minutes used in the last 24 h")
        try:
            path = self.client.predict(json.dumps(texts), self.voice, seed, api_name="/synth")
        except Exception as e:  # gradio_client raises AppError / generic errors for quota and queue issues
            msg = str(e)
            if "quota" in msg.lower() or "GPU task aborted" in msg:
                raise QuotaPause(f"ZeroGPU refused the call: {msg[:200]}")
            raise
        z = np.load(path)
        self.sr = int(z["sr"])
        rows.append([time.time(), float(z["gpu_s"])])
        self.ledger.write_text(json.dumps([r for r in rows if time.time() - r[0] < 86_400]))
        return [z[f"c{i}"].astype(np.float32) / 32767 for i in range(len(texts))]


class LocalBackend:
    """The same model on this machine's GPU (e.g. an HF Job): CLONE_BACKEND=local."""

    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        self.synth, self.sr = clone_backend(cfg)

    def synth_many(self, texts: list[str], seed: int) -> list[np.ndarray]:
        out = []
        for i, t in enumerate(texts):
            self.torch.manual_seed(seed + i)
            out.append(self.synth(t))
        return out


def synthesize_clone(blocks: list[Block], voice: str, speed: float):
    """Same contract as synthesize_kokoro, with a voice cloned by an open-weight model.

    Runs on the voice's private ZeroGPU Space when the profile names one (free within the
    daily allowance), otherwise on a local GPU. Long paragraphs are split into sentence groups
    (long inputs make these models speed up or run on); chunks are rendered in batches, and any
    chunk far off the voice's measured speaking rate (truncated or babbling) is regenerated with
    a new seed, keeping the attempt closest to the expected length. Accepted chunks are cached
    in .tts_cache/, so a render paused by the daily GPU allowance resumes where it stopped."""
    global SAMPLE_RATE, BITRATE
    import hashlib
    cfg = load_voice(voice)
    cps, max_chars = cfg.get("chars_per_sec", 19.0), cfg.get("max_chars", 320)
    if speed != 1.0:
        print("  note: --speed is ignored for the clone engine")
    version = hashlib.sha1(json.dumps({k: cfg.get(k) for k in ("model", "prompt_text", "params", "lora")},
                                      sort_keys=True).encode()).hexdigest()[:10]
    cache = ROOT / ".tts_cache" / f"{voice}-{version}"
    cache.mkdir(parents=True, exist_ok=True)
    key = lambda text: cache / (hashlib.sha1(text.encode()).hexdigest() + ".npy")
    use_space = cfg.get("space") and os.environ.get("CLONE_BACKEND", "space") != "local"
    backend = None

    def chunks(text: str) -> list[str]:
        out, cur = [], ""
        for s in re.split(r"(?<=[.?!:;])\s+", text):
            if cur and len(cur) + len(s) + 1 > max_chars:
                out.append(cur.strip()); cur = ""
            cur += s + " "
        return out + ([cur.strip()] if cur.strip() else [])

    plan = [(b, chunks(b.text)) for b in blocks]
    todo = list(dict.fromkeys(c for _, cs in plan for c in cs if not key(c).exists()))
    total = len({c for _, cs in plan for c in cs})
    print(f"  {total - len(todo)}/{total} chunks cached; rendering {len(todo)} "
          f"on {'ZeroGPU Space ' + cfg['space'] if use_space else 'local GPU'}", flush=True)

    best: dict[str, tuple[float, np.ndarray]] = {}
    batch_chars = int(cfg.get("batch_chars", 1200))
    for attempt in range(4):
        if not todo:
            break
        if backend is None:
            backend = SpaceBackend(cfg, voice, cache) if use_space else LocalBackend(cfg)
        batches, cur = [], []
        for t in todo:
            if cur and sum(map(len, cur)) + len(t) > batch_chars:
                batches.append(cur); cur = []
            cur.append(t)
        batches += [cur] if cur else []
        retry = []
        for bi, batch in enumerate(batches, 1):
            clips = backend.synth_many(batch, seed=1234 + 1000 * attempt)
            for text, a in zip(batch, clips):
                a = trim_silence(a, backend.sr)
                ratio = (len(a) / backend.sr) / max(len(text) / cps, 0.3)
                err = abs(np.log(max(ratio, 1e-3)))
                if text not in best or err < best[text][0]:
                    best[text] = (err, a)
                if 0.55 <= ratio <= 1.8 or attempt == 3:
                    np.save(key(text), np.round(best[text][1] * 32767).astype(np.int16))
                    (cache / "sr").write_text(str(backend.sr))
                else:
                    retry.append(text)
                    print(f"    retry: {ratio:.2f}x expected length for {text[:50]!r}", flush=True)
            done = total - len(todo) + sum(len(b) for b in batches[:bi])
            print(f"  batch {bi}/{len(batches)} (pass {attempt + 1}): {done}/{total} chunks", flush=True)
        todo = retry

    SAMPLE_RATE = int((cache / "sr").read_text())
    BITRATE = "128k" if SAMPLE_RATE > 24_000 else "96k"
    gap = np.zeros(int(SAMPLE_RATE * cfg.get("sentence_gap", 0.12)), dtype=np.float32)
    for b, cs in plan:
        out = []
        for i, c in enumerate(cs):
            out += ([gap] if i else []) + [np.load(key(c)).astype(np.float32) / 32767]
        yield b, (np.concatenate(out) if out else np.zeros(0, dtype=np.float32))


ENGINES = {"kokoro": synthesize_kokoro, "speechify": synthesize_speechify, "clone": synthesize_clone}
EXIT_PAUSED = 75     # render paused by the daily GPU allowance; CI treats this as "try again later"


def paused_exit(gen):
    """Turn a QuotaPause from the clone engine into a clean, resumable exit."""
    try:
        yield from gen
    except QuotaPause as e:
        print(f"\nPAUSED: {e}.\nFinished chunks are cached in .tts_cache/; rerun the same command "
              f"after the allowance resets (24 h after its first use) to continue.", flush=True)
        sys.exit(EXIT_PAUSED)


def encode_mp3(samples: np.ndarray, dest: Path, title: str) -> None:
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 0:
        samples = samples * (PEAK / peak)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
         "-codec:a", "libmp3lame", "-b:a", BITRATE,
         "-metadata", f"title={title}", "-metadata", "artist=Papers, as Audio",
         str(dest)],
        input=samples.astype("<f4").tobytes(), check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", help="lecture id: lectures/<id>.md → audio/<id>.mp3")
    ap.add_argument("--engine", choices=sorted(ENGINES),
                    help="kokoro (free, local; default), speechify (API, $SPEECHIFY_API_KEY), or clone "
                         "(open-weight voice cloning on a GPU; --voice names a profile in voices/); "
                         "defaults to the lecture's `engine` in catalog.json")
    ap.add_argument("--voice", help="Kokoro voice name or Speechify voice id; "
                                    "defaults to the lecture's `voice` in catalog.json")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="re-render even if audio/<id>.mp3 exists")
    ap.add_argument("--dry-run", action="store_true", help="print the narration script and chapters, no audio")
    args = ap.parse_args()

    src = ROOT / "lectures" / f"{args.id}.md"
    if not src.exists():
        sys.exit(f"no transcript at {src}")
    catalog = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
    entry = next((l for l in catalog["lectures"] if l["id"] == args.id), {})
    engine = args.engine or entry.get("engine", "kokoro")
    voice = args.voice or entry.get("voice") or {"kokoro": "af_heart"}.get(engine)
    if not voice:
        sys.exit(f"{engine} needs --voice (or a `voice` field on the catalog entry)")
    out_dir = ROOT / "audio"
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / f"{args.id}.mp3"
    chapters_path = out_dir / f"{args.id}.chapters.json"
    if dest.exists() and not args.force and not args.dry_run:
        sys.exit(f"{dest.relative_to(ROOT)} exists; use --force to re-render")

    blocks = markdown_to_blocks(src.read_text(encoding="utf-8"))
    title = next((b.text for b in blocks if b.level == 1), args.id)
    n_chars = sum(len(b.text) for b in blocks)
    n_chapters = sum(1 for b in blocks if b.level == 2)
    print(f"{src.name}: {len(blocks)} blocks, {n_chars:,} characters, {n_chapters} chapters"
          f"  [{engine} / {voice}]", flush=True)

    if args.dry_run:
        for b in blocks:
            print(("#" * b.level + " " if b.level else "") + b.text)
        return

    silence = lambda secs: np.zeros(int(SAMPLE_RATE * secs), dtype=np.float32)
    pieces: list[np.ndarray] = []
    chapters: list[dict] = []
    t = 0.0
    t0 = time.time()

    for i, (b, audio) in enumerate(paused_exit(ENGINES[engine](blocks, voice, args.speed)), 1):
        if b.level:
            pieces.append(silence(GAP_HEADING)); t += GAP_HEADING
            if b.level == 2:
                chapters.append({"title": b.text.rstrip("."), "start": round(t, 2)})
        pieces.append(audio); t += len(audio) / SAMPLE_RATE
        if len(audio):                   # batched engines yield empty audio for merged blocks
            gap = GAP_AFTER_HEADING if b.level else GAP_PARA
            pieces.append(silence(gap)); t += gap
        if i % 20 == 0 or b.level == 2:
            print(f"  {i:4d}/{len(blocks)}  {t / 60:5.1f} min  "
                  f"({t / max(time.time() - t0, 1e-6):.1f}x real time)"
                  + (f"  ## {b.text[:60]}" if b.level == 2 else ""), flush=True)

    if chapters and chapters[0]["start"] > 20:      # the opening before the first "## " heading
        chapters.insert(0, {"title": "Introduction", "start": 0})

    samples = np.concatenate(pieces)
    encode_mp3(samples, dest, title)
    chapters_path.write_text(json.dumps(chapters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    took = time.time() - t0
    print(f"\n{dest.relative_to(ROOT)}  {dest.stat().st_size / 1e6:.1f} MB  {t / 60:.1f} min of audio  "
          f"{len(chapters)} chapters  generated in {took / 60:.1f} min ({t / took:.1f}x real time)")


if __name__ == "__main__":
    main()
