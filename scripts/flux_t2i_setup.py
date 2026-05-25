#!/usr/bin/env python3
"""Download FLUX.1-dev text-to-image models into ComfyUI (for the `flux` image engine /
character-LoRA frames), then restart ComfyUI so it scans them. Run on the GPU box.

    python scripts/flux_t2i_setup.py

flux1-dev.safetensors + ae.safetensors come from the GATED black-forest-labs/FLUX.1-dev
repo: set HF_TOKEN and accept the license once at
https://huggingface.co/black-forest-labs/FLUX.1-dev . The text encoders are ungated.

After this, in the ComfyUI GUI open a FLUX text-to-image (+ LoRA) template and point:
  Load Diffusion Model -> flux1-dev.safetensors
  DualCLIPLoader       -> t5xxl_fp16 + clip_l   (type: flux)
  Load VAE             -> ae.safetensors
  Load LoRA            -> <character>.safetensors  (e.g. wanwan.safetensors)
Then Export (API) to comfyui/flux_t2i_api.json for the `flux` engine.

Set FLUX_T5_FP8=1 to grab the smaller fp8 T5 (saves ~5GB disk/VRAM, slightly lower quality).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

COMFY = os.environ.get("COMFY", "/content/ComfyUI")
PORT = int(os.environ.get("PORT", "8188"))
HF = os.environ.get("HF_TOKEN")
BFL = "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main"
ENC = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main"
T5 = "t5xxl_fp8_e4m3fn.safetensors" if os.environ.get("FLUX_T5_FP8") else "t5xxl_fp16.safetensors"

# (models subdir, url, gated?, alt subdirs to also check for an existing copy)
FILES = [
    ("diffusion_models", f"{BFL}/flux1-dev.safetensors", True, ("unet",)),
    ("vae", f"{BFL}/ae.safetensors", True, ()),
    ("text_encoders", f"{ENC}/{T5}", False, ("clip",)),
    ("text_encoders", f"{ENC}/clip_l.safetensors", False, ("clip",)),
]


def _present(name: str, dirs: list[str]) -> str | None:
    for d in dirs:
        p = f"{d}/{name}"
        if os.path.isfile(p) and os.path.getsize(p) > 1e8:
            return p
    return None


def main() -> None:
    def dirs_for(sub: str, alt: tuple) -> list[str]:
        return [f"{COMFY}/models/{sub}", *(f"{COMFY}/models/{a}" for a in alt)]

    gated_missing = [u.rsplit("/", 1)[-1] for (s, u, g, alt) in FILES
                     if g and not _present(u.rsplit("/", 1)[-1], dirs_for(s, alt))]
    if gated_missing and not HF:
        raise SystemExit(
            f"[flux] {', '.join(gated_missing)} are GATED — set HF_TOKEN and accept the license at\n"
            "       https://huggingface.co/black-forest-labs/FLUX.1-dev , then rerun.")

    if subprocess.run("command -v aria2c", shell=True).returncode != 0:
        subprocess.run("apt-get -qq install -y aria2", shell=True, check=True)

    for sub, url, gated, alt in FILES:
        d = f"{COMFY}/models/{sub}"
        os.makedirs(d, exist_ok=True)
        name = url.rsplit("/", 1)[-1]
        found = _present(name, dirs_for(sub, alt))
        if found:
            print(f"[skip] {name} ({os.path.getsize(found) / 1e9:.1f} GB present at {found})")
            continue
        print(f"[flux] downloading {name}")
        cmd = ["aria2c", "-x16", "-s16", "-k1M", "-c", "-d", d, "-o", name]
        if HF:
            cmd.append(f"--header=Authorization: Bearer {HF}")
        cmd.append(url)
        subprocess.run(cmd, check=True)

    print("[flux] restarting ComfyUI to scan the new models")
    subprocess.run("pkill -f 'ComfyUI/main.py'", shell=True)
    time.sleep(3)
    logf = open("/content/comfyui.log", "ab")
    subprocess.Popen(
        [sys.executable, f"{COMFY}/main.py", "--listen", "0.0.0.0", "--port", str(PORT)],
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    # wait until ComfyUI is actually listening, so the next stage doesn't race it
    import urllib.request
    print("[flux] waiting for ComfyUI to be ready...")
    for _ in range(90):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=3)
            print("[flux] ComfyUI ready"); break
        except Exception:
            pass
    else:
        print("[flux] ComfyUI 还没就绪，看 /content/comfyui.log")
    print("[flux] done. In the GUI: pick flux1-dev + t5xxl/clip_l + ae + your LoRA, then Queue.")


if __name__ == "__main__":
    main()
