#!/usr/bin/env python3
"""Generate a lecture MP3 with the local Kokoro pipeline.

Replaces the old per-lecture Speechify scripts: no API key, no per-character
billing, no network call. Same Kokoro-82M weights and `af_heart` voice as the
Natural Voice web app, so the whole library sounds like one narrator.

    python generate_lecture.py inference_serving_lecture.txt inference_serving

Writes <slug>_parts/part_NNN.mp3 and <slug>_combined.mp3.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24_000
VOICE = "af_heart"          # grade A, the warmest of the Kokoro voices
PACE = 1.0
PARA_GAP = 0.35             # seconds of silence between paragraphs
PART_TARGET = 300           # seconds of audio per part file


def normalise(text: str) -> str:
    """Tidy pasted text so the reader does not stumble on layout artefacts."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("‘", "'").replace("’", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("—", " - ").replace("–", "-").replace(" ", " ")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate wrapped words
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)          # unwrap soft line breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def synthesize(text: str):
    """Yield (paragraph_text, audio) as each paragraph is rendered."""
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    gap = np.zeros(int(SAMPLE_RATE * PARA_GAP), dtype=np.float32)
    first = True
    for graphemes, _phonemes, audio in pipe(
        text, voice=VOICE, speed=PACE, split_pattern=r"\n{1,}"
    ):
        if audio is None:
            continue
        a = np.asarray(audio, dtype=np.float32)
        if not first:
            a = np.concatenate([gap, a])
        first = False
        yield graphemes, a


def to_mp3(samples: np.ndarray, dest: Path) -> None:
    """Encode via ffmpeg reading raw float32 on stdin — no temp WAV needed."""
    peak = float(np.abs(samples).max())
    if peak > 0:
        samples = samples * (0.891 / peak)          # Kokoro renders quiet
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
         "-codec:a", "libmp3lame", "-b:a", "96k", str(dest)],
        input=samples.astype("<f4").tobytes(), check=True,
    )


def concat(parts: list[Path], dest: Path) -> None:
    listing = dest.parent / "_concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(dest)],
        check=True,
    )
    listing.unlink()


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <text-file> <slug>")
    src, slug = Path(sys.argv[1]), sys.argv[2]
    parts_dir = Path(f"{slug}_parts")
    parts_dir.mkdir(exist_ok=True)
    for old in parts_dir.glob("part_*.mp3"):
        old.unlink()

    text = normalise(src.read_text(encoding="utf-8"))
    print(f"{len(text):,} characters", flush=True)

    t0 = time.time()
    buf: list[np.ndarray] = []
    buffered = 0.0
    total = 0.0
    parts: list[Path] = []

    def flush() -> None:
        nonlocal buf, buffered
        if not buf:
            return
        dest = parts_dir / f"part_{len(parts) + 1:03d}.mp3"
        to_mp3(np.concatenate(buf), dest)
        parts.append(dest)
        print(f"  {dest.name}  {buffered / 60:.1f} min"
              f"  (elapsed {time.time() - t0:.0f}s)", flush=True)
        buf, buffered = [], 0.0

    for i, (_para, audio) in enumerate(synthesize(text), 1):
        secs = len(audio) / SAMPLE_RATE
        buf.append(audio)
        buffered += secs
        total += secs
        if buffered >= PART_TARGET:
            flush()
        if i % 25 == 0:
            print(f"  …{i} paragraphs, {total / 60:.1f} min so far", flush=True)
    flush()

    out = Path(f"{slug}_combined.mp3")
    concat(parts, out)
    took = time.time() - t0
    print(f"\n{out}  {out.stat().st_size / 1e6:.1f} MB"
          f"  {total / 60:.1f} min of audio"
          f"  generated in {took / 60:.1f} min ({total / took:.1f}x real time)")
    print("PARTS", len(parts))


if __name__ == "__main__":
    main()
