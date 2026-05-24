#!/usr/bin/env bash
# Start the ComfyUI server fully detached and wait until it's listening.
# Run on the GPU box (Colab). Frees the terminal so you can then run shotforge.
#
#   bash scripts/comfyui_serve.sh
#   python -m shotforge.generate --project projects/example --engine comfy
set -e

COMFY="${COMFY:-/content/ComfyUI}"
PORT="${PORT:-8188}"
LOG="${LOG:-/content/comfyui.log}"

# setsid: run ComfyUI in its OWN session so it survives even if you Ctrl+C this
# script or close the terminal. --listen 0.0.0.0 serves both localhost
# (shotforge) and a tunnel (GUI).
setsid nohup python "$COMFY/main.py" --listen 0.0.0.0 --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &

echo "[serve] ComfyUI starting in background, log -> $LOG"
echo "[serve] waiting for :$PORT (first start can take ~30-90s)"
echo "[serve] Ctrl+C is SAFE here — it only stops this wait; ComfyUI keeps running"

for _ in $(seq 1 90); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null || true)"
  if [ "$code" = "200" ]; then
    echo
    echo "[serve] ComfyUI up on :$PORT  ->  now run shotforge --engine comfy"
    exit 0
  fi
  printf '.'
  sleep 2
done

echo
echo "[serve] still not up after ~3min; last log lines:"
tail -n 25 "$LOG"
exit 1
