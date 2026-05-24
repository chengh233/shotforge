#!/usr/bin/env bash
# ComfyUI + Wan 2.2 I2V setup — run on the GPU box (Colab A100), NOT the MacBook.
#
# Installs ComfyUI and downloads the Wan 2.2 I2V-A14B model files (~60GB).
# diffusers' Wan I2V produced melting/garbage output; ComfyUI's native Wan
# implementation is the community-proven, high-quality path. See docs/COMFYUI.md.
#
#   bash scripts/comfyui_setup.sh
#   python /content/ComfyUI/main.py --listen 0.0.0.0 --port 8188
set -e

COMFY="${COMFY:-/content/ComfyUI}"

echo "[comfyui] installing into $COMFY"
# Check for main.py, NOT just the dir: a partial/leftover dir (e.g. only models/)
# would otherwise skip the clone and leave ComfyUI's code missing.
if [ ! -f "$COMFY/main.py" ]; then
  echo "[comfyui] cloning ComfyUI (preserving any existing models/)"
  tmp="$(mktemp -d)"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$tmp/ComfyUI"
  mkdir -p "$COMFY"
  cp -rn "$tmp/ComfyUI/." "$COMFY/"   # add code without clobbering downloaded models
  rm -rf "$tmp"
fi
# ComfyUI does NOT pin torch (Colab ships it); this installs its other deps.
pip install -q -r "$COMFY/requirements.txt"

mkdir -p "$COMFY/models/diffusion_models" "$COMFY/models/text_encoders" \
         "$COMFY/models/vae" "$COMFY/models/loras"

DM="$COMFY/models/diffusion_models"
TE="$COMFY/models/text_encoders"
VAE="$COMFY/models/vae"
LO="$COMFY/models/loras"
BASE22="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
BASE21="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"

# Faster downloads: aria2 with many parallel connections (much faster than wget
# for big files). Set HF_TOKEN in the env for higher HF rate limits.
command -v aria2c >/dev/null 2>&1 || { echo "[comfyui] installing aria2"; apt-get -qq update && apt-get -qq install -y aria2; }
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(--header="Authorization: Bearer ${HF_TOKEN}")
dl() {  # dl <url> <target-dir>  — filename = url basename; resumes; 16 connections
  echo "[comfyui] downloading $(basename "$1") -> $2"
  aria2c -x16 -s16 -k1M --continue=true "${AUTH[@]}" -d "$2" "$1"
}

echo "[comfyui] downloading Wan 2.2 I2V models (fp8 experts + lightx2v 4-step LoRAs)"
dl "$BASE22/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" "$DM"
dl "$BASE22/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"  "$DM"
dl "$BASE21/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"              "$TE"
dl "$BASE22/vae/wan_2.1_vae.safetensors"                                       "$VAE"
dl "$BASE22/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"   "$LO"
dl "$BASE22/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"    "$LO"

# Official native Wan 2.2 I2V workflow — drag this file onto the ComfyUI canvas.
wget -c -O /content/wan2_2_i2v_workflow.json \
  "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_14B_i2v.json"

echo
echo "[comfyui] done. Next:"
echo "  1) start server:  python $COMFY/main.py --listen 0.0.0.0 --port 8188"
echo "  2) open the UI from your Mac via a cloudflared tunnel (see docs/COMFYUI.md)"
echo "  3) load /content/wan2_2_i2v_workflow.json, set your start frame + prompt, Queue"
