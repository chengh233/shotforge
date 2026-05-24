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

`setup` installs/starts ComfyUI + Wan models and verifies them (scripts/colab_setup.py).
The other stages just call the matching Python module.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

USAGE = "usage: python run.py <setup|genref|frames|video|stitch|dub|subs|post> [project|character] [extra args]"


def run(*cmd: str) -> None:
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), check=True, cwd=HERE)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(USAGE)
    stage, rest = sys.argv[1], sys.argv[2:]

    if stage == "setup":
        run(PY, "scripts/colab_setup.py")
        return

    if not rest:
        sys.exit(f"stage '{stage}' needs a project path, e.g. projects/example")
    project, extra = rest[0], rest[1:]

    if stage == "genref":  # `project` here is a character id (characters/<id>)
        run(PY, "-m", "shotforge.genref", "--character", project, *extra)
    elif stage == "frames":
        run(PY, "-m", "shotforge.frames", "--project", project, *extra)
    elif stage == "video":
        run(PY, "-m", "shotforge.generate", "--project", project, "--engine", "comfy", *extra)
    elif stage == "stitch":
        run(PY, "-m", "tools.stitch", "--project", project)
    elif stage == "dub":
        run(PY, "-m", "tools.dub", "--project", project, *extra)
    elif stage == "subs":
        run(PY, "-m", "tools.subtitle", "--project", project)
    elif stage == "post":
        run(PY, "-m", "tools.post", "--project", project, *extra)
    else:
        sys.exit(f"unknown stage: {stage}\n{USAGE}")


if __name__ == "__main__":
    main()
