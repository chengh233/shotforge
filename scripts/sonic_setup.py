#!/usr/bin/env python3
"""Install ComfyUI_Sonic (audio-driven talking head) + its models, for the `sonic` talk engine.

    python scripts/sonic_setup.py
    # then in the GUI: build a Sonic graph (LoadImage + LoadAudio -> Sonic -> video output),
    #   Export (API) to comfyui/sonic_api.json, then:
    python run.py talk projects/<p>

Sonic = ONE pass from a portrait + audio to a talking head with coordinated head motion +
lip sync. Models:
  - Sonic weights (LeonJoe13/Sonic) + whisper-tiny  -> ComfyUI/models/sonic/
  - SVD svd_xt.safetensors                          -> ComfyUI/models/checkpoints/
    (GATED: set HF_TOKEN and accept the license once at
     https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)

The custom node's requirements install into ComfyUI's own python (this script's python).
Heavy + the most involved install here — rerun-safe; tweak as errors surface.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

COMFY = os.environ.get("COMFY", "/content/ComfyUI")
PORT = int(os.environ.get("PORT", "8188"))
HF = os.environ.get("HF_TOKEN")
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI_Sonic")
SONIC_DIR = os.path.join(COMFY, "models", "sonic")
CKPT_DIR = os.path.join(COMFY, "models", "checkpoints")


def sh(*cmd: str, **kw) -> None:
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), check=True, **kw)


def _hf_snapshot(repo: str, local_dir: str, sentinel: str) -> None:
    if os.path.isfile(os.path.join(local_dir, sentinel)):
        print(f"[skip] {repo} present")
        return
    print(f"[sonic] downloading {repo} -> {local_dir}")
    sh(sys.executable, "-c",
       f"from huggingface_hub import snapshot_download as d; d({repo!r}, local_dir={local_dir!r})")


def main() -> None:
    if not os.path.isdir(NODE):
        sh("git", "clone", "--depth", "1", "https://github.com/smthemex/ComfyUI_Sonic", NODE)
    req = os.path.join(NODE, "requirements.txt")
    if os.path.isfile(req):
        sh(sys.executable, "-m", "pip", "install", "-r", req)   # into ComfyUI's python
    sh(sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub")

    os.makedirs(SONIC_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    _hf_snapshot("LeonJoe13/Sonic", SONIC_DIR, "unet.pth")
    _hf_snapshot("openai/whisper-tiny", os.path.join(SONIC_DIR, "whisper-tiny"), "model.safetensors")

    svd = os.path.join(CKPT_DIR, "svd_xt.safetensors")
    if not os.path.isfile(svd):
        if not HF:
            raise SystemExit(
                "[sonic] svd_xt.safetensors is GATED — set HF_TOKEN and accept the license at\n"
                "        https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt , then rerun.")
        print("[sonic] downloading svd_xt.safetensors (gated)")
        sh(sys.executable, "-c",
           "from huggingface_hub import hf_hub_download; import shutil; "
           "p=hf_hub_download('stabilityai/stable-video-diffusion-img2vid-xt','svd_xt.safetensors',"
           f"token={HF!r}); shutil.copy(p, {svd!r})")

    print("[sonic] restarting ComfyUI to load the new node + models")
    subprocess.run("pkill -f 'ComfyUI/main.py'", shell=True)
    time.sleep(3)
    logf = open("/content/comfyui.log", "ab")
    subprocess.Popen(
        [sys.executable, f"{COMFY}/main.py", "--listen", "0.0.0.0", "--port", str(PORT)],
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    print("[sonic] waiting for ComfyUI to be ready...")
    for _ in range(120):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=3)
            print("[sonic] ComfyUI ready"); break
        except Exception:
            pass
    else:
        print("[sonic] ComfyUI 还没就绪，看 /content/comfyui.log")
    print("[sonic] done. In the GUI: build LoadImage + LoadAudio -> Sonic -> video output,")
    print("        Export (API) -> comfyui/sonic_api.json, then: python run.py talk projects/<p>")


if __name__ == "__main__":
    main()
