#!/usr/bin/env python3
"""Staged pipeline runner (Python). Run any stage on its own so you can check
quality between steps. WHERE each stage runs (to minimize Colab GPU time): see
docs/STAGES.md.

  COLAB (GPU):  python run.py setup
                python run.py video  projects/example [--shot s1]
  MAC   (CPU):  python run.py dub    projects/example [--voice zh-CN-YunxiNeural]
                python run.py subs   projects/example
                python run.py post   projects/example [--music bgm.mp3]
                python run.py stitch projects/example   # silent concat only

`setup` (Colab, one command): ComfyUI + Wan models (colab_setup.py) + Qwen-Image models
(qwen_setup.py) + a background cloudflared tunnel that prints the ComfyUI GUI URL
(tunnel.py). The other stages just call the matching Python module.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

USAGE = ("usage: python run.py <setup|genref|train|frames|verify|manga|video|dub|subs|lipsync|post|stitch> "
         "[project|character] [extra args]\n"
         "  frames/video/lipsync go through shotforge.pipeline (compose + pluggable engines)\n"
         "  setup [--comfy-only|--flux]: full (ComfyUI+Wan+Qwen), ComfyUI-only, or ComfyUI+FLUX T2I models\n"
         "  set COMFY_URL=<cloudflared url> to drive a remote Colab ComfyUI from your Mac")


def run(*cmd: str) -> None:
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), check=True, cwd=HERE)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(USAGE)
    stage, rest = sys.argv[1], sys.argv[2:]

    if stage == "setup":
        # Pick the model set; the tunnel runs at the end in every case.
        #   --comfy-only : just install + serve ComfyUI (no model downloads)
        #   --flux       : ComfyUI + FLUX.1-dev text-to-image models (for character-LoRA
        #                  frames) — skips the heavy Wan/Qwen downloads
        #   (default)    : full — ComfyUI + Wan I2V + Qwen image models
        flags = sys.argv[2:]
        if "--comfy-only" in flags:
            run(PY, "scripts/colab_setup.py", "--comfy-only")
        elif "--flux" in flags:
            run(PY, "scripts/colab_setup.py", "--comfy-only")   # ComfyUI itself
            run(PY, "scripts/flux_t2i_setup.py")                # + FLUX.1-dev T2I models
        else:
            run(PY, "scripts/colab_setup.py")   # ComfyUI + Wan models + serve
            run(PY, "scripts/qwen_setup.py")    # Qwen-Image models (default image engine)
        run(PY, "scripts/tunnel.py")            # cloudflared tunnel (background) -> prints the GUI URL
        return

    if not rest:
        sys.exit(f"stage '{stage}' needs a project path, e.g. projects/example")
    project, extra = rest[0], rest[1:]

    if stage == "genref":  # `project` here is a character id (characters/<id>)
        run(PY, "-m", "shotforge.genref", "--character", project, *extra)
    elif stage == "train":  # `project` here is a character id — train its FLUX LoRA (GPU)
        run(PY, "scripts/train_lora.py", "--character", project, *extra)
    elif stage in ("frames", "video", "lipsync"):   # compose + pluggable engines
        run(PY, "-m", "shotforge.pipeline", stage, "--project", project, *extra)
    elif stage == "manga":  # assemble frames into a 条漫/storyboard page (read it as stills)
        run(PY, "-m", "tools.manga", "--project", project, *extra)
    elif stage == "verify":  # VLM checks each frame vs its intent (auto QA gate)
        run(PY, "-m", "tools.verify", "--project", project, *extra)
    elif stage == "stitch":
        run(PY, "-m", "tools.stitch", "--project", project)
    elif stage in ("dub", "voice"):
        run(PY, "-m", "tools.dub", "--project", project, *extra)
    elif stage == "subs":
        run(PY, "-m", "tools.subtitle", "--project", project)
    elif stage == "post":
        run(PY, "-m", "tools.post", "--project", project, *extra)
    else:
        sys.exit(f"unknown stage: {stage}\n{USAGE}")


if __name__ == "__main__":
    main()
