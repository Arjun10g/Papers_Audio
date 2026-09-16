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
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 24_000
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

def synthesize(blocks: list[Block], voice: str, speed: float):
    """Yield (block, audio float32) in order; silence gaps are added by the caller."""
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    for b in blocks:
        chunks = []
        for _graphemes, _phonemes, audio in pipe(b.text, voice=voice, speed=speed, split_pattern=r"\n+"):
            if audio is not None:
                chunks.append(np.asarray(audio, dtype=np.float32))
        yield b, (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32))


def encode_mp3(samples: np.ndarray, dest: Path, title: str) -> None:
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 0:
        samples = samples * (PEAK / peak)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
         "-codec:a", "libmp3lame", "-b:a", "96k",
         "-metadata", f"title={title}", "-metadata", "artist=Papers, as Audio",
         str(dest)],
        input=samples.astype("<f4").tobytes(), check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", help="lecture id: lectures/<id>.md → audio/<id>.mp3")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="re-render even if audio/<id>.mp3 exists")
    ap.add_argument("--dry-run", action="store_true", help="print the narration script and chapters, no audio")
    args = ap.parse_args()

    src = ROOT / "lectures" / f"{args.id}.md"
    if not src.exists():
        sys.exit(f"no transcript at {src}")
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
    print(f"{src.name}: {len(blocks)} blocks, {n_chars:,} characters, {n_chapters} chapters", flush=True)

    if args.dry_run:
        for b in blocks:
            print(("#" * b.level + " " if b.level else "") + b.text)
        return

    silence = lambda secs: np.zeros(int(SAMPLE_RATE * secs), dtype=np.float32)
    pieces: list[np.ndarray] = []
    chapters: list[dict] = []
    t = 0.0
    t0 = time.time()

    for i, (b, audio) in enumerate(synthesize(blocks, args.voice, args.speed), 1):
        if b.level:
            pieces.append(silence(GAP_HEADING)); t += GAP_HEADING
            if b.level == 2:
                chapters.append({"title": b.text.rstrip("."), "start": round(t, 2)})
        pieces.append(audio); t += len(audio) / SAMPLE_RATE
        gap = GAP_AFTER_HEADING if b.level else GAP_PARA
        pieces.append(silence(gap)); t += gap
        if i % 20 == 0 or b.level == 2:
            print(f"  {i:4d}/{len(blocks)}  {t / 60:5.1f} min  "
                  f"({t / max(time.time() - t0, 1e-6):.1f}x real time)"
                  + (f"  ## {b.text[:60]}" if b.level == 2 else ""), flush=True)

    samples = np.concatenate(pieces)
    encode_mp3(samples, dest, title)
    chapters_path.write_text(json.dumps(chapters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    took = time.time() - t0
    print(f"\n{dest.relative_to(ROOT)}  {dest.stat().st_size / 1e6:.1f} MB  {t / 60:.1f} min of audio  "
          f"{len(chapters)} chapters  generated in {took / 60:.1f} min ({t / took:.1f}x real time)")


if __name__ == "__main__":
    main()
