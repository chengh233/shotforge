#!/usr/bin/env python3
"""Download Wan 2.2 **Text-to-Video** models into ComfyUI (for the wan-t2v engine),
then restart ComfyUI so it scans them. Run on the GPU box after colab_setup.py.

    python scripts/wan_t2v_setup.py

T2V is a SEPARATE model set from I2V: it needs the two T2V experts (high/low noise).
The umt5 text-encoder + Wan VAE are shared with I2V and skipped if already present.

After this, get a T2V **workflow**: in the ComfyUI GUI open the "Wan 2.2 14B Text to
Video" template, Export (API), and save it to comfyui/wan_t2v_api.json. Then:
    python run.py video projects/vlog
HF_TOKEN (optional) speeds downloads.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

COMFY = os.environ.get("COMFY", "/content/ComfyUI")
PORT = int(os.environ.get("PORT", "8188"))
HF = os.environ.get("HF_TOKEN")
BASE = "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"

# (subfolder, url). The two T2V experts are T2V-specific; umt5 + vae are shared
# with I2V (skip-if-present handles them).
FILES = [
    ("diffusion_models", f"{BASE}/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
    ("diffusion_models", f"{BASE}/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
    ("text_encoders", f"{BASE}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    ("vae", f"{BASE}/vae/wan_2.1_vae.safetensors"),
    # lightx2v 4-step speed LoRAs (the T2V template's "Missing Models")
    ("loras", f"{BASE}/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"),
    ("loras", f"{BASE}/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"),
]


def main() -> None:
    if subprocess.run("command -v aria2c", shell=True).returncode != 0:
        subprocess.run("apt-get -qq install -y aria2", shell=True, check=True)

    for sub, url in FILES:
        d = f"{COMFY}/models/{sub}"
        os.makedirs(d, exist_ok=True)
        name = url.rsplit("/", 1)[-1]
        path = f"{d}/{name}"
        if os.path.isfile(path) and os.path.getsize(path) > 1e8:
            print(f"[skip] {name} ({os.path.getsize(path) / 1e9:.1f} GB present)")
            continue
        print(f"[wan-t2v] downloading {name}")
        cmd = ["aria2c", "-x16", "-s16", "-k1M", "-c", "-d", d, "-o", name]
        if HF:
            cmd.append(f"--header=Authorization: Bearer {HF}")
        cmd.append(url)
        subprocess.run(cmd, check=True)

    print("[wan-t2v] restarting ComfyUI to scan the new models")
    subprocess.run("pkill -f 'ComfyUI/main.py'", shell=True)
    time.sleep(3)
    logf = open("/content/comfyui.log", "ab")
    subprocess.Popen(
        [sys.executable, f"{COMFY}/main.py", "--listen", "0.0.0.0", "--port", str(PORT)],
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    print("[wan-t2v] done. Next: export the Wan 2.2 T2V template as API -> comfyui/wan_t2v_api.json,\n"
          "          then: python run.py video projects/vlog")


if __name__ == "__main__":
    main()
