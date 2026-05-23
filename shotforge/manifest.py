"""Data model + project loader for shotforge.

A "project" is one short-drama script: a folder under projects/ holding a
project.yaml manifest and a frames/ directory of starting-frame PNGs. The
manifest's top-level ``model:`` field selects a backend (see backends.py),
whose constraints drive the frame-count and dimension snapping below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml

from .backends import get_backend


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
    model: str          # backend name from project.yaml `model:` (e.g. "ltx")
    fps: int
    shots: list[Shot]


def frames_for(seconds: float, fps: int, quantum: int = 8) -> int:
    """Number of frames to render for a clip.

    Most I2V models require ``num_frames == quantum * N + 1`` (LTX: 8, Wan: 4).
    We pick the value of that form closest to ``seconds * fps``, with a floor of
    one full quantum. ``quantum`` comes from the project's model backend.
    """
    target = max(quantum + 1, round(seconds * fps))
    k = max(1, round((target - 1) / quantum))
    return quantum * k + 1


def snap_dim(x: int, multiple: int = 32) -> int:
    """Snap a dimension to the nearest valid multiple (LTX: 32, Wan: 16).

    Video dims must divide the model's ``dim_multiple`` or the pipeline errors
    or produces garbage. ``multiple`` comes from the project's model backend.
    """
    return max(multiple, round(x / multiple) * multiple)


def load_project(path: str) -> Project:
    """Load ``<path>/project.yaml`` and resolve each shot.

    The ``model:`` field selects a backend; per-shot values fall back to the
    optional ``defaults`` block, then to the backend's defaults. Frame paths are
    joined with the project dir and width/height are snapped to the backend's
    dimension multiple.
    """
    manifest = os.path.join(path, "project.yaml")
    with open(manifest, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    backend = get_backend(data.get("model"))
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
                width=snap_dim(int(merged.get("width", backend.default_width)), backend.dim_multiple),
                height=snap_dim(int(merged.get("height", backend.default_height)), backend.dim_multiple),
                steps=int(merged.get("steps", backend.default_steps)),
                seed=int(merged.get("seed", Shot.seed)),
                negative=str(merged.get("negative", Shot.negative)),
            )
        )

    name = str(data.get("project") or os.path.basename(os.path.abspath(path)))
    fps = int(data.get("fps", 24))
    return Project(root=path, name=name, model=backend.name, fps=fps, shots=shots)
