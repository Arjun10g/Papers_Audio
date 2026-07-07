import base64
import ast
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


API_URL = os.getenv("SPEECHIFY_API_URL", "https://api.sws.speechify.com/v1/audio/speech")
VOICE_ID = os.getenv("SPEECHIFY_VOICE_ID", "2c91daee-2a3a-40d9-9a8c-db8d7ab17bd6")
API_KEY = os.getenv("SPEECHIFY_API_KEY")

if not API_KEY:
    # Reuse the credential already present in the existing local generation scripts
    # without importing them, since older scripts depend on third-party requests.
    match = re.search(r'^API_KEY\s*=\s*(".*?")', Path("recurrent_depth_speech.py").read_text(), re.M)
    if match:
        API_KEY = ast.literal_eval(match.group(1))
    else:
        raise RuntimeError("Set SPEECHIFY_API_KEY or add an API_KEY to recurrent_depth_speech.py")

OUTPUT_DIR = Path("ouro_implementation_parts")
TEXT_FILE = Path("ouro-implementation-lecture.md")
OUTPUT_FILE = Path("ouro_implementation_combined.mp3")


def split_text(text, max_len=1900):
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def generate_audio(chunk, output_path):
    payload = {
        "input": chunk,
        "voice_id": VOICE_ID,
        "audio_format": "mp3",
        "language": "en-US",
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Speechify request failed: {exc.code} {detail}") from exc

    audio_bytes = base64.b64decode(json.loads(response_body)["audio_data"])
    output_path.write_bytes(audio_bytes)


def combine_with_ffmpeg(parts_dir, output_file):
    files = sorted(parts_dir.glob("part_*.mp3"))
    if not files:
        raise RuntimeError("No generated MP3 parts found")

    list_file = parts_dir / "filelist.txt"
    with list_file.open("w") as f:
        for mp3 in files:
            f.write(f"file '{mp3.name}'\n")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        list_file.unlink(missing_ok=True)

    print(f"Combined {len(files)} parts into {output_file}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    text = TEXT_FILE.read_text()
    chunks = split_text(text)
    print(f"Split into {len(chunks)} parts")

    for i, chunk in enumerate(chunks, 1):
        output_path = OUTPUT_DIR / f"part_{i:03d}.mp3"
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Skipping existing {output_path}")
            continue
        generate_audio(chunk, output_path)
        print(f"Saved {output_path}")

    combine_with_ffmpeg(OUTPUT_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()
