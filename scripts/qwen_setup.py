#!/usr/bin/env python3
"""Download Qwen-Image models into ComfyUI (for the `qwen` image engine), then
restart ComfyUI so it scans them. Run on the GPU box after colab_setup.py.

    python scripts/qwen_setup.py

Then get a workflow: in the ComfyUI GUI open the "Qwen-Image" (text-to-image) or
"Qwen-Image-Edit" (reference-conditioned) template, Export (API), save it to
comfyui/qwen_image_api.json. Then set a project's `engines: image: qwen` and run
`python run.py frames <project>`. HF_TOKEN (optional) speeds downloads.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

COMFY = os.environ.get("COMFY", "/content/ComfyUI")
PORT = int(os.environ.get("PORT", "8188"))
HF = os.environ.get("HF_TOKEN")
BASE = "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files"

FILES = [
    ("diffusion_models", f"{BASE}/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"),
    ("text_encoders", f"{BASE}/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("vae", f"{BASE}/vae/qwen_image_vae.safetensors"),
    # Lightning 4-step speed LoRA (the 2512 template's "Missing Models"); from lightx2v
    ("loras", "https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"),
    # Qwen-Image-Edit 2511 (latest/best edit model) + its 4-step Lightning LoRA — for FLF2V end frames
    ("diffusion_models", "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"),
    ("loras", "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"),
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
        print(f"[qwen] downloading {name}")
        cmd = ["aria2c", "-x16", "-s16", "-k1M", "-c", "-d", d, "-o", name]
        if HF:
            cmd.append(f"--header=Authorization: Bearer {HF}")
        cmd.append(url)
        subprocess.run(cmd, check=True)

    print("[qwen] restarting ComfyUI to scan the new models")
    subprocess.run("pkill -f 'ComfyUI/main.py'", shell=True)
    time.sleep(3)
    logf = open("/content/comfyui.log", "ab")
    subprocess.Popen(
        [sys.executable, f"{COMFY}/main.py", "--listen", "0.0.0.0", "--port", str(PORT)],
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    print("[qwen] done. Next: export the Qwen-Image template as API -> comfyui/qwen_image_api.json,\n"
          "        set a project `engines: image: qwen`, then: python run.py frames <project>")


if __name__ == "__main__":
    main()
