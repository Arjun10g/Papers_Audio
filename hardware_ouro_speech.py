import re
import requests
import base64
import subprocess
from pathlib import Path

API_KEY = "__TXVgPwAz8t-BETN4l8tBj3OdUiVEN5SCuZ0CQobXw="
API_URL = "https://api.sws.speechify.com/v1/audio/speech"
VOICE_ID = "2c91daee-2a3a-40d9-9a8c-db8d7ab17bd6"
OUTPUT_DIR = Path("hardware_ouro_parts")
TEXT_FILE = Path("hardware-ouro-lecture.md")


def strip_markdown(text):
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_text(text, max_len=1900):
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_len:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks


def generate_audio(chunk, output_path):
    payload = {
        "input": chunk,
        "voice_id": VOICE_ID,
        "audio_format": "mp3",
        "language": "en-US",
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(API_URL, json=payload, headers=headers)
    resp.raise_for_status()
    audio_bytes = base64.b64decode(resp.json()["audio_data"])
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def combine_with_ffmpeg(parts_dir, output_file):
    files = sorted(parts_dir.glob("part_*.mp3"))
    list_file = parts_dir / "filelist.txt"
    with open(list_file, "w") as f:
        for mp3 in files:
            f.write(f"file '{mp3.name}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output_file)],
        check=True, capture_output=True,
    )
    list_file.unlink()
    print(f"Combined {len(files)} parts into {output_file}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = TEXT_FILE.read_text()
    text = strip_markdown(raw)
    chunks = split_text(text)
    print(f"Split into {len(chunks)} parts")

    for i, chunk in enumerate(chunks, 1):
        output_path = OUTPUT_DIR / f"part_{i:03d}.mp3"
        try:
            generate_audio(chunk, output_path)
            print(f"Saved {output_path}")
        except requests.HTTPError as e:
            print(f"Error on part {i}: {e.response.status_code} - {e.response.text}")
            continue

    combine_with_ffmpeg(OUTPUT_DIR, Path("hardware_ouro_combined.mp3"))


if __name__ == "__main__":
    main()
