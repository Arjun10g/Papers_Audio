# Papers, as Audio

Long-form technical lectures (statistics, machine learning, LLM inference)
narrated as audio, delivered as an installable phone app.

- **App:** https://arjun10g.github.io/Papers_Audio/ — open it in Chrome on
  Android and tap *Install app* (or the browser's *Add to Home screen*).
- **Audio + transcripts:** Hugging Face dataset
  [`arjun10g/papers-audio`](https://huggingface.co/datasets/arjun10g/papers-audio).
  The app streams from there and reads `library.json` on every launch, so a
  newly published lecture appears on the phone without reinstalling anything.

## How it fits together

```
catalog.json            ← the ordered list of lectures (the only file you edit by hand)
lectures/<id>.md        ← transcript / narration source, one per lecture
audio/<id>.mp3          ← narrated audio (git-ignored; lives on Hugging Face)
audio/<id>.chapters.json← section markers written by the narrator
app/                    ← the PWA, deployed to GitHub Pages by CI
tools/generate_lecture.py   markdown → mp3 with Kokoro-82M (free, local)
tools/publish.py            sync audio/transcripts → HF, build library.json
.github/workflows/publish.yml   push to main → narrate missing → publish → deploy
```

`library.json` (on the dataset root) is the contract between the pipeline and
the app: id, title, series, duration, size, sha256, transcript path, chapter
markers, artwork, and a `base` URL pinned to the dataset commit so cached
audio never goes stale.

## Add a lecture

1. Write `lectures/<id>.md`. Use `## ` headings for sections — each one
   becomes a chapter marker you can jump to in the app. Write for the ear
   (see `docs/audio-writing-prompt.md`).
2. Append an entry to `catalog.json`:
   ```json
   { "id": "<id>", "title": "…", "series": "…",
     "bg": "https://images.unsplash.com/…?w=1920&q=80", "published": "2026-09-16" }
   ```
3. `git push`. The workflow narrates it with Kokoro on the CI runner (10–30
   min for a long lecture), uploads the mp3 + transcript, republishes
   `library.json`, and redeploys the app. The phone picks it up on next launch
   with a *NEW* badge.

To narrate locally instead (faster on an M-series Mac), then push:

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install kokoro soundfile numpy -r tools/requirements.txt   # brew install espeak-ng ffmpeg
python tools/generate_lecture.py <id>           # → audio/<id>.mp3 + chapters
HF_TOKEN=hf_… python tools/publish.py          # upload + library.json
```

CI publishes with the `HF_TOKEN` repository secret (a write token for
`arjun10g`). Re-rendering an existing lecture: `generate_lecture.py <id>
--force`, then publish — the new sha256 gives it a new pinned URL.

## The app

Vanilla HTML/CSS/JS in `app/`, no build step. Offline downloads go through the
Cache API; the service worker answers `Range` requests from the cache so
seeking works offline. Lock-screen controls use the Media Session API. Playback
position, speed, theme and downloads persist per device. `APP_VERSION` in
`app/config.js` and `app/sw.js` is stamped with the commit SHA on deploy, which
is what triggers the in-app "Update ready" prompt.

Run it locally with any static server, e.g. `python3 -m http.server -d app 8000`.
