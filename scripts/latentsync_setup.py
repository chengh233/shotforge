#!/usr/bin/env python3
"""Install LatentSync (ByteDance) for the `latentsync` lip-sync engine, then print the
LIPSYNC_CMD to export. Run on the GPU box.

    python scripts/latentsync_setup.py
    export LIPSYNC_CMD='...'           # the line it prints
    python run.py lipsync projects/<p>  # re-times each speaking shot's mouth to its voice

LatentSync re-times a talking face's mouth to its voice audio. It's trained on REAL
faces, so it's a great fit for photorealistic characters (like wanwan). Heavy, one-time:
a CLEAN venv (LatentSync installs its own pinned torch, isolated from the box's torch)
plus ~5GB of checkpoints. Like the training step, it may need a tweak on a given box —
the venv + command are printed so you can rerun/adjust by hand.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

LS = os.environ.get("LATENTSYNC", "/content/LatentSync")
VENV = os.environ.get("LATENTSYNC_VENV", "/content/latentsync-venv")
VENV_PY = os.path.join(VENV, "bin", "python")


def sh(*cmd: str, **kw) -> None:
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), check=True, **kw)


def _ensure_pip(py: str) -> None:
    """Bootstrap pip in the venv (Colab's python ships no ensurepip -> get-pip.py)."""
    if subprocess.run([py, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        return
    try:
        sh(py, "-m", "ensurepip", "--upgrade")
        return
    except subprocess.CalledProcessError:
        pass
    import tempfile
    import urllib.request
    g = os.path.join(tempfile.gettempdir(), "get-pip.py")
    print("[latentsync] bootstrapping pip via get-pip.py")
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", g)
    sh(py, g)


def _python310() -> str:
    """LatentSync's pinned deps (mediapipe 0.10.11, decord 0.6.0, insightface …) only have
    wheels for py3.10. This box is py3.12, so provision python3.10 (deadsnakes) for the venv."""
    p = shutil.which("python3.10")
    if p:
        return p
    print("[latentsync] box is py3.12 but LatentSync's deps target 3.10 — installing python3.10")
    sh("bash", "-c",
       "apt-get -qq update && apt-get -qq install -y software-properties-common && "
       "add-apt-repository -y ppa:deadsnakes/ppa && apt-get -qq update && "
       "apt-get -qq install -y python3.10 python3.10-venv")
    p = shutil.which("python3.10")
    if not p:
        raise SystemExit("[latentsync] couldn't install python3.10 — install it manually and rerun")
    return p


def _is_py310(venv_py: str) -> bool:
    if not os.path.isfile(venv_py):
        return False
    out = subprocess.run([venv_py, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                         capture_output=True, text=True)
    return out.stdout.strip() == "3.10"


def main() -> None:
    if not os.path.isfile(os.path.join(LS, "scripts", "inference.py")):
        sh("git", "clone", "--depth", "1", "https://github.com/bytedance/LatentSync", LS)
    # (Re)create the venv with python3.10 if it's missing or built with the wrong python.
    # Clean venv (no --system-site-packages): LatentSync pins its own torch/xformers.
    # --without-pip then get-pip.py, so we don't depend on ensurepip being present.
    if not _is_py310(VENV_PY):
        py310 = _python310()
        if os.path.isdir(VENV):
            shutil.rmtree(VENV, ignore_errors=True)
        sh(py310, "-m", "venv", "--without-pip", VENV)
    _ensure_pip(VENV_PY)
    sh(VENV_PY, "-m", "pip", "install", "-q", "-U", "pip")
    # not -q: this pulls torch+cuda (several GB) and looks "stuck" when silent
    sh(VENV_PY, "-m", "pip", "install", "-r", os.path.join(LS, "requirements.txt"))

    ckpt = os.path.join(LS, "checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    if not os.path.isfile(os.path.join(ckpt, "latentsync_unet.pt")):
        print("[latentsync] downloading checkpoints (~5GB) from ByteDance/LatentSync-1.5")
        sh(VENV_PY, "-m", "pip", "install", "-q", "-U", "huggingface_hub")
        sh(VENV_PY, "-c",
           "from huggingface_hub import snapshot_download as d; "
           f"d('ByteDance/LatentSync-1.5', local_dir={ckpt!r}, "
           "allow_patterns=['latentsync_unet.pt', 'whisper/tiny.pt'])")

    # LIPSYNC_CMD template the `latentsync` engine fills with {video} {audio} {out}.
    cmd = ("cd %s && %s -m scripts.inference "
           "--unet_config_path configs/unet/stage2.yaml "
           "--inference_ckpt_path checkpoints/latentsync_unet.pt "
           "--inference_steps 20 --guidance_scale 1.5 "
           "--video_path {video} --audio_path {audio} --video_out_path {out}") % (LS, VENV_PY)
    print("\n[latentsync] ✅ done. Export this (then `python run.py lipsync <project>`):\n")
    print(f"export LIPSYNC_CMD='{cmd}'\n")


if __name__ == "__main__":
    main()
