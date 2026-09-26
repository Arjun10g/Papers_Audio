#!/bin/bash
# Common job bootstrap: system audio deps, python deps ($PIP), then fetch and run a script from the private repo.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq ffmpeg sox libsndfile1 git build-essential >/dev/null
pip -q install --root-user-action=ignore huggingface_hub soundfile $PIP 2>&1 | tail -2
SCRIPT=$(python -c "from huggingface_hub import hf_hub_download as d; print(d('arjun10g/papers-audio-voice','jobs/'+'$1',repo_type='dataset',force_download=True))")
shift
python "$SCRIPT" "$@"
