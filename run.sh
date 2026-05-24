#!/usr/bin/env bash
# Staged pipeline — run any stage on its own so you can check quality between
# steps. WHERE each stage runs (to minimize Colab GPU time): see docs/STAGES.md.
#
#   COLAB (GPU):  bash run.sh setup
#                 bash run.sh serve
#                 bash run.sh video  projects/example [--shot s1]
#   MAC   (CPU):  bash run.sh dub    projects/example [--voice zh-CN-YunxiNeural]
#                 bash run.sh subs   projects/example
#                 bash run.sh post   projects/example [--music bgm.mp3]
#                 bash run.sh stitch projects/example   # silent concat only
set -e

stage="${1:?usage: bash run.sh <setup|serve|video|stitch|dub|subs|post> <project> [args]}"
shift || true

case "$stage" in
  setup)  bash scripts/comfyui_setup.sh ;;
  serve)  bash scripts/comfyui_serve.sh ;;
  video)  python -m shotforge.generate --project "$1" --engine comfy "${@:2}" ;;
  stitch) python -m tools.stitch    --project "$1" ;;
  dub)    python -m tools.dub       --project "$1" "${@:2}" ;;
  subs)   python -m tools.subtitle  --project "$1" ;;
  post)   python -m tools.post      --project "$1" "${@:2}" ;;
  *) echo "unknown stage: $stage (setup|serve|video|stitch|dub|subs|post)"; exit 1 ;;
esac
