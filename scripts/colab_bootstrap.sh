#!/usr/bin/env bash
# One-command setup on a FRESH Colab GPU box: deps + ComfyUI + Wan models +
# server + cloudflared. Idempotent (auto-skips what's already present).
#
# Empty-Colab one-liner (paste into a Colab cell):
#   !git clone https://github.com/chengh233/shotforge /content/shotforge 2>/dev/null; \
#    cd /content/shotforge && git pull -q && bash scripts/colab_bootstrap.sh
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== 1/4 light deps (comfy engine + post; no torch) =="
pip install -q pyyaml requests edge-tts

echo "== 2/4 ComfyUI + Wan models (downloads only what's missing) =="
bash scripts/comfyui_setup.sh

echo "== 3/4 start ComfyUI server =="
bash scripts/comfyui_serve.sh

echo "== 4/4 cloudflared (for GUI / driving from your Mac) =="
command -v cloudflared >/dev/null 2>&1 || {
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared; }

cat <<'NEXT'

[bootstrap] done — ComfyUI is up on :8188.
Next:
  • Render here:           python -m shotforge.generate --project projects/example --engine comfy --shot s1
  • GUI / remote API URL:  cloudflared tunnel --url http://localhost:8188   (prints a https://...trycloudflare.com)
NEXT
