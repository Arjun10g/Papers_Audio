---
title: Teller TTS
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Private Teller narrator voice for Papers, as Audio
---

Private ZeroGPU Space: VoxCPM2 + the Teller LoRA. Called by `tools/generate_lecture.py`
in https://github.com/Arjun10g/Papers_Audio. Voice data is read from the private dataset
`arjun10g/papers-audio-voice` using this Space's `HF_TOKEN` secret.
