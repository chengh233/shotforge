"""Scene library — reusable "sets" you place stories in.

The mirror of the character library: where the characters/ folder answers *who*
(an actor: appearance, reference image, voice), the scenes/ folder answers
*where* (a set: one reference image of the location + a description). A project
picks one with ``scene: <id>`` in its project.yaml; frame generation then feeds
that scene reference alongside the character reference so every shot — including
ones without the protagonist — keeps the same location, art style and aspect.
A set lives in ``scenes/<id>/scene.yaml`` (sibling of ``characters/``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


@dataclass
class Scene:
    id: str
    name: str = ""
    description: str = ""     # English description of the set (for prompts)
    ref: str = ""             # absolute path to ONE reference image of the location


def scenes_dir() -> str:
    """The repo-level ``scenes/`` directory (sibling of ``characters/``)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenes")


def load_scene(scene: str) -> Scene:
    """Load a scene by id (``scenes/<id>/``) or by an explicit dir path."""
    sdir = scene if os.path.isdir(scene) else os.path.join(scenes_dir(), scene)
    manifest = os.path.join(sdir, "scene.yaml")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"scene not found: {manifest}")
    with open(manifest, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    ref = data.get("ref")
    return Scene(
        id=os.path.basename(os.path.normpath(sdir)),
        name=str(data.get("name", "")),
        description=str(data.get("description", "")),
        ref=os.path.join(sdir, str(ref)) if ref else "",
    )


def list_scenes() -> list[str]:
    d = scenes_dir()
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, "scene.yaml")))
