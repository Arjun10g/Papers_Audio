#!/bin/bash
# usage: waitjobs.sh <job id>...   polls until every job is terminal, then prints the stages
: "${HF_TOKEN:?set HF_TOKEN (write token for arjun10g)}"
stage() { hf jobs inspect "$1" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d.get("status",{}).get("stage"))' 2>/dev/null; }
while true; do
  all=1; line=""
  for id in "$@"; do s=$(stage "$id"); line="$line ${id:0:8}=$s"
    case "$s" in COMPLETED|ERROR|CANCELED|DELETED) ;; *) all=0;; esac
  done
  [ $all = 1 ] && { echo "FINAL:$line"; break; }
  sleep 30
done
