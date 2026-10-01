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
brew install espeak-ng ffmpeg
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install kokoro soundfile numpy -r tools/requirements.txt
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
python tools/generate_lecture.py <id>           # → audio/<id>.mp3 + chapters
HF_TOKEN=hf_… python tools/publish.py          # upload + library.json
```

CI publishes with the `HF_TOKEN` repository secret (a write token for
`arjun10g`). Re-rendering an existing lecture: `generate_lecture.py <id>
--force`, then publish — the new sha256 gives it a new pinned URL.

### Voices

Kokoro (`af_heart`) is the default: free and local. For a Speechify voice,
put it on the catalog entry and the narrator (locally or in CI) uses it:

```json
{ "id": "expert-prefetch", "engine": "speechify",
  "voice": "1d6165c1-8b4c-455e-ac30-edd3784606a5", ... }
```

Speechify needs `SPEECHIFY_API_KEY` in the environment (it is also a
repository secret for CI). List the account's voice ids with
`curl -H "Authorization: Bearer $SPEECHIFY_API_KEY" https://api.sws.speechify.com/v1/voices`.
`--engine` / `--voice` on the command line override the catalog.

### The Teller voice, without Speechify

`"engine": "clone", "voice": "teller"` narrates with an open-weight clone of
the Speechify "Teller" voice: **VoxCPM2** (Apache-2.0, 48 kHz) plus a **LoRA
fine-tuned on about 19 minutes of Teller narration**. The profile (prompt clip,
transcript, LoRA weights) lives in the **private** HF dataset
`arjun10g/papers-audio-voice` under `voices/teller/`, never in this repo.

**It costs nothing.** The model runs on the private ZeroGPU Space
[`arjun10g/teller-tts`](https://huggingface.co/spaces/arjun10g/teller-tts)
(source in `tools/voice_clone/space/`), inside the PRO account's free daily GPU
allowance of 40 minutes. Rendering needs about 0.9 GPU-seconds per second of
audio, so a 45-minute lecture fits in one day's allowance; a longer one pauses
and finishes the next day. So adding a Teller lecture is the same as any other:
write it, add `"engine": "clone", "voice": "teller"` to its catalog entry, push.
CI renders it through the Space, and a daily scheduled run resumes any render
the allowance paused, then publishes.

Guard rails: finished chunks are cached (`.tts_cache/`, kept in the Actions
cache), and the client stops at 36 free minutes per rolling 24 hours
(`ZEROGPU_BUDGET_MIN`). This matters because past the allowance Hugging Face
bills *prepaid credits* if the account holds any; with a $0 credit balance the
Space just refuses. Long paragraphs are split into sentence groups, and any
chunk far off Teller's measured speaking rate is regenerated.

To render locally instead (your Mac only coordinates; the Space does the work):
`HF_TOKEN=… python tools/generate_lecture.py <id>` (needs `pip install
gradio_client numpy`). `CLONE_BACKEND=local` runs the model on a local GPU,
e.g. inside a paid HF Job (`tools/voice_clone/stage3_render.py`).

How it was chosen (`tools/voice_clone/`, all run as HF Jobs): real Teller
narration was cut into prompt, held-out and training clips; two models
(VoxCPM2, Qwen3-TTS) read the held-out sentences, and each system was scored
for speaker similarity against real Teller (ECAPA), word error (Whisper
large-v3) and pace. Held-out results:

| system | similarity | WER | pace |
|---|---|---|---|
| real Teller vs itself (ceiling) | 0.881 | 2.6% | 1.00 |
| **VoxCPM2 + Teller LoRA (1,000 steps)** | **0.835** | **2.6%** | 1.06 |
| VoxCPM2 zero-shot, 25 s reference | 0.789 | 2.7% | 1.01 |
| Qwen3-TTS zero-shot, 25 s reference | 0.782 | 2.2% | 1.05 |

### Password-protected lectures

Some lectures should not be readable by anyone else. Everything in this repo,
the dataset and the app is public, so protection is encryption, not a login
screen: only ciphertext is ever published, and the app decrypts on the device
after the password is entered (with an option to remember it on that device).

```bash
# 1. text + a private catalog entry, both in the git-ignored private/ folder
private/lectures/<id>.md
private/catalog.json        # {"lectures": [{"id", "title", "series", "bg", "engine", "voice"}]}
# 2. narrate as usual (renders into the git-ignored audio/)
HF_TOKEN=… python tools/generate_lecture.py <id>
# 3. encrypt + upload the ciphertext, write locked.json (safe to commit)
LECTURE_PASSWORD=… HF_TOKEN=… python tools/lock_lecture.py <id>
# 4. publish + push as usual; publish.py appends locked.json to library.json
```

Crypto: PBKDF2-HMAC-SHA-256 (600,000 iterations, random salt) → AES-256-GCM
for the audio, the transcript, and the title/series/artwork/chapters. The
library shows "Locked lecture" with only duration and size in the clear. The
ciphertext is publicly downloadable, so the protection is only as strong as the
password: a short one can be guessed offline.

For a locked lecture that must be rendered in GitHub Actions, use
`python tools/locked_source.py encrypt <id> --github-secret`. This commits
only an encrypted source envelope under `pending_locked/`; run the
`Publish locked lecture` workflow after pushing it. The workflow decrypts in
the runner, renders with the private Teller profile, uploads encrypted audio
and transcript to Hugging Face, commits `locked.json`, removes the pending
envelope, and deploys the updated library. A daily run retries a render paused
by the free GPU allowance. The `LECTURE_PASSWORD` Actions secret can be
removed after the workflow succeeds.

## The app

Vanilla HTML/CSS/JS in `app/`, no build step. Offline downloads go through the
Cache API; the service worker answers `Range` requests from the cache so
seeking works offline. Lock-screen controls use the Media Session API. Playback
position, speed, theme and downloads persist per device. `APP_VERSION` in
`app/config.js` and `app/sw.js` is stamped with the commit SHA on deploy, which
is what triggers the in-app "Update ready" prompt.

Run it locally with any static server, e.g. `python3 -m http.server -d app 8000`.
