#!/usr/bin/env python3
"""Download Flux.1 Kontext [dev] models into ComfyUI (for the `frames` stage),
then restart ComfyUI so it scans them. Run on the GPU box after colab_setup.py.

    python scripts/flux_setup.py

Flux Kontext keeps a reference character consistent across new scenes — used by
shotforge's `frames` stage to make all of a project's starting frames at once.
HF_TOKEN (optional) speeds downloads.
"""
from __future__ import annotations

import os
import subprocess
import time

COMFY = os.environ.get("COMFY", "/content/ComfyUI")
PORT = int(os.environ.get("PORT", "8188"))
HF = os.environ.get("HF_TOKEN")

# (subfolder, url) — verified from the ComfyUI Flux Kontext docs.
FILES = [
    ("diffusion_models", "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors"),
    ("vae", "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors"),
    ("text_encoders", "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"),
    ("text_encoders", "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors"),
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
        print(f"[flux] downloading {name}")
        cmd = ["aria2c", "-x16", "-s16", "-k1M", "-c", "-d", d, "-o", name]
        if HF:
            cmd.append(f"--header=Authorization: Bearer {HF}")
        cmd.append(url)
        subprocess.run(cmd, check=True)

    # restart ComfyUI so it scans the new models (detached, won't hang the cell)
    print("[flux] restarting ComfyUI to scan the new models")
    subprocess.run("pkill -f 'ComfyUI/main.py'", shell=True)
    time.sleep(3)
    logf = open("/content/comfyui.log", "ab")
    import sys
    subprocess.Popen(
        [sys.executable, f"{COMFY}/main.py", "--listen", "0.0.0.0", "--port", str(PORT)],
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    print("[flux] done. Wait ~20s for ComfyUI, then: python run.py frames projects/lasttram")


if __name__ == "__main__":
    main()
