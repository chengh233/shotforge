#!/usr/bin/env python3
"""One-shot, self-verifying ComfyUI + Wan 2.2 I2V setup for a fresh Colab GPU box.

Why Python (not bash) on Colab: it can (1) kill a stale ComfyUI and restart so it
re-scans freshly downloaded models, and (2) query ComfyUI's /object_info to
*confirm* every model is actually visible before exiting — so the recurring
"Value not in list" error can't slip through. Idempotent: re-running skips files
already downloaded.

    !cd /content/shotforge && python scripts/colab_setup.py

HF_TOKEN (optional, for faster downloads) is read from the environment.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

COMFY = os.environ.get("COMFY", "/content/ComfyUI")
PORT = int(os.environ.get("PORT", "8188"))
HF = os.environ.get("HF_TOKEN")

B22 = "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
B21 = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"
# (model subdir, url, min_bytes) — min size guards against truncated downloads.
MODELS = [
    ("diffusion_models", f"{B22}/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 10e9),
    ("diffusion_models", f"{B22}/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 10e9),
    ("text_encoders", f"{B21}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 4e9),
    ("vae", f"{B22}/vae/wan_2.1_vae.safetensors", 1e8),
    ("loras", f"{B22}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 1e8),
    ("loras", f"{B22}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", 1e8),
]


def sh(cmd: str) -> None:
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def clone_comfyui() -> None:
    if os.path.isfile(f"{COMFY}/main.py"):
        print("[setup] ComfyUI present")
        return
    print("[setup] cloning ComfyUI (preserving any existing models/)")
    sh("rm -rf /tmp/_comfy && git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /tmp/_comfy")
    os.makedirs(COMFY, exist_ok=True)
    sh(f"cp -rn /tmp/_comfy/. {COMFY}/ && rm -rf /tmp/_comfy")


def install_deps() -> None:
    sh(f"pip install -q -r {COMFY}/requirements.txt")
    sh("pip install -q pyyaml requests edge-tts")
    if subprocess.run("command -v aria2c", shell=True).returncode != 0:
        sh("apt-get -qq install -y aria2")


def download_models() -> None:
    for sub, url, minb in MODELS:
        d = f"{COMFY}/models/{sub}"
        os.makedirs(d, exist_ok=True)
        name = url.rsplit("/", 1)[-1]
        path = f"{d}/{name}"
        if os.path.isfile(path) and os.path.getsize(path) >= minb:
            print(f"[skip] {name} ({os.path.getsize(path) / 1e9:.1f} GB present)")
            continue
        print(f"[download] {name}")
        cmd = ["aria2c", "-x16", "-s16", "-k1M", "-c", "-d", d, "-o", name]
        if HF:
            cmd.append(f"--header=Authorization: Bearer {HF}")
        cmd.append(url)
        subprocess.run(cmd, check=True)


def restart_comfyui() -> None:
    print("[setup] (re)starting ComfyUI so it re-scans the models")
    subprocess.run("pkill -f 'ComfyUI/main.py'", shell=True)
    time.sleep(3)
    # Fully detach: own session + stdout/stderr to a log file + stdin from
    # /dev/null. Otherwise the launching notebook cell keeps spinning forever,
    # holding the stdout pipe open even after this script is done.
    logf = open("/content/comfyui.log", "ab")
    subprocess.Popen(
        [sys.executable, f"{COMFY}/main.py", "--listen", "0.0.0.0", "--port", str(PORT)],
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_until_models_visible(timeout: float = 300.0) -> None:
    """Block until ComfyUI answers AND lists every model we downloaded."""
    wanted = {url.rsplit("/", 1)[-1] for _, url, _ in MODELS}
    url = f"http://127.0.0.1:{PORT}/object_info"
    deadline = time.time() + timeout
    missing = wanted
    while time.time() < deadline:
        try:
            info = json.load(urllib.request.urlopen(url, timeout=10))
        except Exception:
            time.sleep(2)
            continue
        seen: set[str] = set()
        for node in info.values():
            req = node.get("input", {}).get("required", {})
            for cfg in req.values():
                if isinstance(cfg, list) and cfg and isinstance(cfg[0], list):
                    seen.update(x for x in cfg[0] if isinstance(x, str))
        missing = wanted - seen
        if not missing:
            print(f"[setup] ✅ ComfyUI up on :{PORT} — all {len(wanted)} models visible")
            print(f"[setup] next: python -m shotforge.generate --project projects/example --engine comfy --shot s1")
            return
        time.sleep(2)
    print(f"[setup] ⚠️ ComfyUI up but these models aren't visible: {sorted(missing)}")
    print("[setup] last log lines:")
    sh("tail -n 25 /content/comfyui.log || true")
    sys.exit(1)


if __name__ == "__main__":
    clone_comfyui()
    install_deps()
    download_models()
    restart_comfyui()
    wait_until_models_visible()
