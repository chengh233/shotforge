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
if [ ! -d "$COMFY" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
fi
# ComfyUI does NOT pin torch (Colab ships it); this installs its other deps.
pip install -q -r "$COMFY/requirements.txt"

mkdir -p "$COMFY/models/diffusion_models" "$COMFY/models/text_encoders" "$COMFY/models/vae"

DM="$COMFY/models/diffusion_models"
TE="$COMFY/models/text_encoders"
VAE="$COMFY/models/vae"
BASE22="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
BASE21="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"

echo "[comfyui] downloading Wan 2.2 I2V-A14B models (~60GB; wget -c resumes on retry)"
# Two 14B experts (high-noise + low-noise) — Wan 2.2 is a MoE; ComfyUI swaps them.
# fp8 variants exist in the same folder (replace fp16 -> fp8_scaled) for less
# disk/VRAM; the A100 supports fp8 fine.
wget -c -P "$DM"  "$BASE22/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
wget -c -P "$DM"  "$BASE22/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
wget -c -P "$TE"  "$BASE21/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
wget -c -P "$VAE" "$BASE22/vae/wan_2.1_vae.safetensors"

# Official native Wan 2.2 I2V workflow — drag this file onto the ComfyUI canvas.
wget -c -O /content/wan2_2_i2v_workflow.json \
  "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_14B_i2v.json"

echo
echo "[comfyui] done. Next:"
echo "  1) start server:  python $COMFY/main.py --listen 0.0.0.0 --port 8188"
echo "  2) open the UI from your Mac via a cloudflared tunnel (see docs/COMFYUI.md)"
echo "  3) load /content/wan2_2_i2v_workflow.json, set your start frame + prompt, Queue"
