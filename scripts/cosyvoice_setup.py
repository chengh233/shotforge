#!/usr/bin/env python3
"""Install CosyVoice (FunAudioLLM) for a natural Chinese TTS voice — replaces edge-tts.
Uses CosyVoice-300M-SFT's preset speaker '中文女' (no reference audio needed).

    python scripts/cosyvoice_setup.py
    # then in the project.yaml: engines: { voice: cosyvoice }
    python run.py dub projects/<p>

One-time, on the GPU box: clones CosyVoice into a py3.10 venv (its own torch — LatentSync's
deps and this box's nightly torch would clash), installs requirements, downloads the
CosyVoice-300M-SFT model. Heavy; like the other installs it may need a tweak — the venv +
model dir are printed so you can rerun/adjust.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

CV = os.environ.get("COSYVOICE", "/content/CosyVoice")
VENV = os.environ.get("COSYVOICE_VENV", "/content/cosyvoice-venv")
VENV_PY = os.path.join(VENV, "bin", "python")
MODEL = os.path.join(CV, "pretrained_models", "CosyVoice-300M-SFT")


def sh(*cmd: str, **kw) -> None:
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), check=True, **kw)


def _ensure_pip(py: str) -> None:
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
    print("[cosyvoice] bootstrapping pip via get-pip.py")
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", g)
    sh(py, g)


def _python310() -> str:
    p = shutil.which("python3.10")
    if p:
        return p
    print("[cosyvoice] installing python3.10 (CosyVoice's deps target 3.10)")
    sh("bash", "-c",
       "apt-get -qq update && apt-get -qq install -y software-properties-common && "
       "add-apt-repository -y ppa:deadsnakes/ppa && apt-get -qq update && "
       "apt-get -qq install -y python3.10 python3.10-venv")
    p = shutil.which("python3.10")
    if not p:
        raise SystemExit("[cosyvoice] couldn't install python3.10 — install it manually and rerun")
    return p


def _is_py310(venv_py: str) -> bool:
    if not os.path.isfile(venv_py):
        return False
    out = subprocess.run([venv_py, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
                         capture_output=True, text=True)
    return out.stdout.strip() == "3.10"


def main() -> None:
    if not os.path.isfile(os.path.join(CV, "cosyvoice", "cli", "cosyvoice.py")):
        sh("git", "clone", "--recursive", "https://github.com/FunAudioLLM/CosyVoice", CV)
    if not _is_py310(VENV_PY):
        py310 = _python310()
        if os.path.isdir(VENV):
            shutil.rmtree(VENV, ignore_errors=True)
        sh(py310, "-m", "venv", "--without-pip", VENV)
    _ensure_pip(VENV_PY)
    sh(VENV_PY, "-m", "pip", "install", "-U", "pip")
    # openai-whisper==20231117 fails to build its wheel here; the latest builds fine and
    # works for CosyVoice's tokenizer. Loosen that pin into a patched requirements file.
    req = os.path.join(CV, "requirements.txt")
    text = open(req, encoding="utf-8").read().replace("openai-whisper==20231117", "openai-whisper")
    patched = os.path.join(CV, "requirements.shotforge.txt")
    open(patched, "w", encoding="utf-8").write(text)
    # not -q: this pulls torch (looks "stuck" when silent)
    sh(VENV_PY, "-m", "pip", "install", "-r", patched)
    # lightning/Matcha-TTS call pkg_resources.declare_namespace, but setuptools>=81 (which
    # get-pip installs) dropped pkg_resources -> pin setuptools back so it exists.
    sh(VENV_PY, "-m", "pip", "install", "setuptools<81")

    if not (os.path.isdir(MODEL) and os.listdir(MODEL)):
        print("[cosyvoice] downloading CosyVoice-300M-SFT from ModelScope")
        sh(VENV_PY, "-m", "pip", "install", "-q", "-U", "modelscope")
        sh(VENV_PY, "-c",
           "from modelscope import snapshot_download as d; "
           f"d('iic/CosyVoice-300M-SFT', local_dir={MODEL!r})")

    print("\n[cosyvoice] ✅ done. In your project.yaml set:\n    engines:\n      voice: cosyvoice")
    print("then:  python run.py dub projects/<project>\n")


if __name__ == "__main__":
    main()
