"""Data model + project loader for shotforge.

A "project" is one short-drama script: a folder under projects/ holding a
project.yaml manifest and a frames/ directory of starting-frame PNGs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class Shot:
    id: str
    frame: str
    prompt: str
    seconds: float = 5.0
    width: int = 480
    height: int = 832
    steps: int = 40
    seed: int = 0
    negative: str = ""


@dataclass
class Project:
    root: str
    name: str
    fps: int
    shots: list[Shot]


def frames_for(seconds: float, fps: int) -> int:
    """Number of frames to render for a clip.

    LTX-Video requires ``num_frames == 8 * N + 1``. We pick the multiple of 8
    (plus one) closest to ``seconds * fps``, with a sane floor.
    """
    n = max(9, round(seconds * fps))
    k = max(1, round((n - 1) / 8))
    return 8 * k + 1


def snap32(x: int) -> int:
    """Snap a dimension to the nearest multiple of 32 (video dims must divide 32)."""
    return max(32, round(x / 32) * 32)


def load_project(path: str) -> Project:
    """Load ``<path>/project.yaml`` and resolve each shot.

    Per-shot values fall back to the optional ``defaults`` block, then to the
    ``Shot`` dataclass defaults. Frame paths are joined with the project dir and
    width/height are snapped to a multiple of 32.
    """
    manifest = os.path.join(path, "project.yaml")
    with open(manifest, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    defaults: dict[str, Any] = data.get("defaults") or {}

    shots: list[Shot] = []
    for raw in data.get("shots") or []:
        merged: dict[str, Any] = {**defaults, **raw}
        shots.append(
            Shot(
                id=str(merged["id"]),
                frame=os.path.join(path, str(merged["frame"])),
                prompt=str(merged.get("prompt", "")),
                seconds=float(merged.get("seconds", Shot.seconds)),
                width=snap32(int(merged.get("width", Shot.width))),
                height=snap32(int(merged.get("height", Shot.height))),
                steps=int(merged.get("steps", Shot.steps)),
                seed=int(merged.get("seed", Shot.seed)),
                negative=str(merged.get("negative", Shot.negative)),
            )
        )

    name = str(data.get("project") or os.path.basename(os.path.abspath(path)))
    fps = int(data.get("fps", 24))
    return Project(root=path, name=name, fps=fps, shots=shots)
