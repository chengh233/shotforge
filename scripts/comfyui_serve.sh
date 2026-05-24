#!/usr/bin/env bash
# Start the ComfyUI server in the background and wait until it's listening.
# Run on the GPU box (Colab). Frees the terminal so you can then run shotforge.
#
#   bash scripts/comfyui_serve.sh
#   python -m shotforge.generate --project projects/example --engine comfy
set -e

COMFY="${COMFY:-/content/ComfyUI}"
PORT="${PORT:-8188}"
LOG="${LOG:-/content/comfyui.log}"

# --listen 0.0.0.0 so it serves both localhost (shotforge) and a tunnel (GUI).
nohup python "$COMFY/main.py" --listen 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
echo "[serve] ComfyUI starting (pid $!), log -> $LOG"

for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || true)"
  if [ "$code" = "200" ]; then
    echo "[serve] ComfyUI up on :$PORT"
    exit 0
  fi
  sleep 2
done

echo "[serve] ComfyUI not up after ~2min; last log lines:"
tail -n 20 "$LOG"
exit 1
