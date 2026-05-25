#!/usr/bin/env python3
"""Train a character LoRA on FLUX.1-dev with ai-toolkit (ostris) — the most proven
FLUX character-LoRA path. Reads characters/<id>/dataset/ and writes the LoRA to
ComfyUI/models/loras/<id>.safetensors (+ a repo copy for backup).

    python scripts/train_lora.py --character mira --trigger mira
    python run.py train mira --trigger mira

Prereqs (one-time, GPU box):
  - HF_TOKEN env, and accept the FLUX.1-dev license once at
    https://huggingface.co/black-forest-labs/FLUX.1-dev  (ai-toolkit downloads it).
ai-toolkit runs in its own venv (reusing system torch). This is the heaviest step
and the one most likely to need a tweak on a given image; the config it writes +
the command it runs are printed so you can rerun/adjust by hand.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFY", "/content/ComfyUI")
AI_TOOLKIT = os.environ.get("AI_TOOLKIT", "/content/ai-toolkit")
VENV = os.environ.get("LORA_VENV", "/content/lora-venv")
VENV_PY = os.path.join(VENV, "bin", "python")
_IMG = (".png", ".jpg", ".jpeg", ".webp")

_CONFIG = """\
job: extension
config:
  name: {name}
  process:
    - type: sd_trainer
      training_folder: {out}
      device: cuda:0
      network: {{ type: lora, linear: {rank}, linear_alpha: {rank} }}
      save: {{ dtype: float16, save_every: {steps}, max_step_saves_to_keep: 1 }}
      datasets:
        - folder_path: {dataset}
          caption_ext: txt
          caption_dropout_rate: 0.05
          resolution: [768, 1024]
      train:
        batch_size: 1
        steps: {steps}
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer: adamw8bit
        lr: 1e-4
        dtype: bf16
      model:
        name_or_path: black-forest-labs/FLUX.1-dev
        is_flux: true
        quantize: true
meta: {{ name: {name} }}
"""


def sh(*cmd: str, **kw) -> None:
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), check=True, **kw)


def _ensure_pip(venv_py: str) -> None:
    """Make `venv_py -m pip` usable. Debian/Colab's split python often ships without
    ensurepip, so a venv it creates has no pip at all ("No module named pip", then
    "No module named ensurepip"). Try ensurepip, then fall back to get-pip.py."""
    if subprocess.run([venv_py, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        return
    try:
        sh(venv_py, "-m", "ensurepip", "--upgrade")
        return
    except subprocess.CalledProcessError:
        pass
    import tempfile
    import urllib.request
    getpip = os.path.join(tempfile.gettempdir(), "get-pip.py")
    print("[train] no pip/ensurepip in venv — bootstrapping via get-pip.py")
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", getpip)
    sh(venv_py, getpip)


def _ensure_torchaudio(venv_py: str) -> None:
    """ai-toolkit imports torchaudio at startup; Colab's system torch (seen via
    --system-site-packages) may not ship it. Install a torchaudio matching the venv's
    torch (same version + CUDA build), --no-deps so it can't drag in a different torch."""
    if subprocess.run([venv_py, "-c", "import torchaudio"], capture_output=True).returncode == 0:
        return
    info = subprocess.run(
        [venv_py, "-c", "import torch;print(torch.__version__.split('+')[0]);print(torch.version.cuda or '')"],
        capture_output=True, text=True)
    parts = info.stdout.split()
    if not parts:
        sh(venv_py, "-m", "pip", "install", "-q", "torchaudio")  # no torch to match — let pip pick
        return
    ver = parts[0]
    cu = parts[1] if len(parts) > 1 else ""
    idx = f"https://download.pytorch.org/whl/cu{cu.replace('.', '')}" if cu else "https://download.pytorch.org/whl/cpu"
    # Exact match first; nightly/preview torch often has no matching torchaudio release yet,
    # so fall back to the newest torchaudio on the same CUDA index (--no-deps keeps torch put).
    for spec in (f"torchaudio=={ver}", "torchaudio"):
        try:
            print(f"[train] installing {spec} from {idx} (--no-deps) for ai-toolkit")
            sh(venv_py, "-m", "pip", "install", "-q", spec, "--index-url", idx, "--no-deps")
            return
        except subprocess.CalledProcessError:
            continue
    raise SystemExit(f"[train] no torchaudio on {idx} matching torch {ver} — install one by hand")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a FLUX character LoRA with ai-toolkit.")
    ap.add_argument("--character", required=True, help="character id (characters/<id>)")
    ap.add_argument("--trigger", default=None, help="trigger word (default the id); also written as captions if missing")
    ap.add_argument("--dataset", default=None, help="dataset dir (default characters/<id>/dataset)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()

    trigger = a.trigger or a.character
    dataset = a.dataset or os.path.join(HERE, "characters", a.character, "dataset")
    imgs = [p for p in glob.glob(os.path.join(dataset, "*")) if p.lower().endswith(_IMG)]
    if not imgs:
        raise SystemExit(f"[train] no images in {dataset} — run tools.dataset first.")
    if not os.environ.get("HF_TOKEN"):
        print("[train] WARNING: no $HF_TOKEN — FLUX.1-dev download will fail. Set it + accept the license on HF.")
    # ensure every image has a caption (trigger word)
    for p in imgs:
        cap = os.path.splitext(p)[0] + ".txt"
        if not os.path.isfile(cap):
            open(cap, "w", encoding="utf-8").write(trigger)

    if not os.path.isfile(os.path.join(AI_TOOLKIT, "run.py")):
        sh("git", "clone", "--depth", "1", "https://github.com/ostris/ai-toolkit", AI_TOOLKIT)
    if not os.path.isfile(VENV_PY):
        sh(sys.executable, "-m", "venv", "--system-site-packages", VENV)
    # Guarantee pip exists *inside* the venv first — a half-built venv from an earlier run
    # can lack it ("No module named pip"); ensurepip bootstraps it. Then (re)install deps
    # every run (notably python-dotenv, which ai-toolkit's run.py imports). Cheap if satisfied.
    _ensure_pip(VENV_PY)
    sh(VENV_PY, "-m", "pip", "install", "-q", "-U", "pip")
    sh(VENV_PY, "-m", "pip", "install", "-q", "-r", os.path.join(AI_TOOLKIT, "requirements.txt"))
    sh(VENV_PY, "-m", "pip", "install", "-q", "python-dotenv")
    _ensure_torchaudio(VENV_PY)

    out_dir = os.path.join("/content", f"lora_out_{a.character}")
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, f"{a.character}.yaml")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        fh.write(_CONFIG.format(name=a.character, out=out_dir, dataset=dataset, steps=a.steps, rank=a.rank))
    print(f"[train] config -> {cfg_path}  (trigger={trigger!r}, {len(imgs)} imgs, {a.steps} steps)")
    sh(VENV_PY, "run.py", cfg_path, cwd=AI_TOOLKIT)

    produced = max(glob.glob(os.path.join(out_dir, "**", "*.safetensors"), recursive=True),
                   key=os.path.getmtime, default=None)
    if not produced:
        raise SystemExit(f"[train] no .safetensors produced in {out_dir}")
    loras = os.path.join(COMFY, "models", "loras")
    os.makedirs(loras, exist_ok=True)
    shutil.copy(produced, os.path.join(loras, f"{a.character}.safetensors"))
    shutil.copy(produced, os.path.join(HERE, "characters", a.character, f"{a.character}.safetensors"))
    print(f"[ok] LoRA -> {loras}/{a.character}.safetensors  (trigger word: {trigger!r})")
    print("[next] use it with the FLUX image engine; back up the repo copy (Colab is ephemeral).")


if __name__ == "__main__":
    main()
