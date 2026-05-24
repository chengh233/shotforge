"""Data model + project loader for shotforge.

A "project" is one short-drama script: ``projects/<name>/project.yaml`` + a
``frames/`` dir + ``out/``. The project only *references* asset-library elements
(cast / scene / style …) and selects engines; it embeds no content. Everything
is resolved here into a ``Project`` the composer + engines consume.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from .backends import get_backend
from .element import Element, load_element


@dataclass
class Shot:
    id: str
    frame: str
    prompt: str = ""          # composed Wan motion prompt (camera + action + style suffix)
    seconds: float = 5.0
    width: int = 480
    height: int = 832
    steps: int = 40
    seed: int = 0
    negative: str = ""
    dialogue: str = ""        # spoken line / narration for this shot (TTS + subtitles)
    frame_prompt: str = ""    # image CONTENT for this shot's frame (a.k.a. `content`)
    camera: str = ""          # camera-move id (cameras.py)
    base: str = ""            # generate this frame from another shot's frame (chaining; keeps seat/pose)
    subjects: list[str] = field(default_factory=list)  # cast roles IN this frame ([] = extras/scenery)
    speaker: str = ""         # cast role whose dialogue/voice this shot carries
    still: bool = False       # frame-only master/blocking shot: generated + usable as `base`, NOT animated


@dataclass
class Project:
    root: str
    name: str
    model: str
    fps: int
    shots: list[Shot]
    engines: dict[str, str] = field(default_factory=dict)   # stage -> engine name
    # cast (the actors)
    cast_map: dict[str, str] = field(default_factory=dict)          # role -> character id
    cast_elements: dict[str, Element] = field(default_factory=dict)  # role -> Element
    # back-compat single-character view (primary cast member)
    character: str = ""
    character_ref: str = ""
    cast: str = ""
    voice: str = ""
    # the set
    scene: str = ""
    scene_ref: str = ""
    scene_desc: str = ""
    scene_el: Element | None = None
    # the look
    style: str = ""
    style_positive: str = ""
    style_negative: str = ""
    style_video_suffix: str = ""
    style_el: Element | None = None

    def voice_for(self, role: str) -> str:
        """TTS voice for a cast role (falls back to the primary voice)."""
        el = self.cast_elements.get(role)
        return (el.voice if el else "") or self.voice


def frames_for(seconds: float, fps: int, quantum: int = 8) -> int:
    target = max(quantum + 1, round(seconds * fps))
    k = max(1, round((target - 1) / quantum))
    return quantum * k + 1


def snap_dim(x: int, multiple: int = 32) -> int:
    return max(multiple, round(x / multiple) * multiple)


def _cast_map(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if raw:
        return {"protagonist": str(raw)}
    return {}


def load_project(path: str) -> Project:
    manifest = os.path.join(path, "project.yaml")
    with open(manifest, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    backend = get_backend(data.get("model"))
    defaults: dict[str, Any] = data.get("defaults") or {}
    engines = {str(k): str(v) for k, v in (data.get("engines") or {}).items()}

    # cast (actors) — a map role->id (or a single id -> {protagonist: id})
    cast_map = _cast_map(data.get("cast"))
    cast_elements = {role: load_element("character", cid) for role, cid in cast_map.items()}
    primary_role = next(iter(cast_map), "")
    primary = cast_elements.get(primary_role)

    # the set + the look (resolved once)
    scene = str(data.get("scene", "") or "")
    scene_el = load_element("scene", scene) if scene else None
    style = str(data.get("style", "") or "")
    style_el = load_element("style", style) if style else None
    style_video_suffix = style_el.motion if style_el else ""

    cam: dict[str, str] | None = None  # camera vocab, lazily loaded

    shots: list[Shot] = []
    for raw in data.get("shots") or []:
        merged: dict[str, Any] = {**defaults, **raw}
        # subjects: cast roles in frame. Back-compat: fall back to use_ref bool.
        subjects = merged.get("subjects")
        if subjects is None:
            subjects = [primary_role] if (merged.get("use_ref", True) and primary_role) else []
        subjects = [str(s) for s in subjects]
        speaker = str(merged.get("speaker", "") or (subjects[0] if subjects else primary_role))

        # motion prompt = <camera move> + <action> + <style video suffix>
        camera_id = str(merged.get("camera", "") or "")
        action = str(merged.get("prompt", merged.get("action", "")))
        if camera_id:
            if cam is None:
                from .cameras import cameras as _load_cameras
                cam = _load_cameras()
            move = cam.get(camera_id, "")
            prompt = "，".join(p for p in (move, action.strip(), style_video_suffix) if p)
        else:
            prompt = action

        shots.append(Shot(
            id=str(merged["id"]),
            frame=os.path.join(path, str(merged["frame"])),
            prompt=prompt,
            seconds=float(merged.get("seconds", Shot.seconds)),
            width=snap_dim(int(merged.get("width", backend.default_width)), backend.dim_multiple),
            height=snap_dim(int(merged.get("height", backend.default_height)), backend.dim_multiple),
            steps=int(merged.get("steps", backend.default_steps)),
            seed=int(merged.get("seed", Shot.seed)),
            negative=str(merged.get("negative", Shot.negative)),
            dialogue=str(merged.get("dialogue", Shot.dialogue)),
            frame_prompt=str(merged.get("content", merged.get("frame_prompt", ""))),
            camera=camera_id,
            base=str(merged.get("base", "") or ""),
            subjects=subjects,
            speaker=speaker,
            still=bool(merged.get("still", False)),
        ))

    name = str(data.get("project") or data.get("title") or os.path.basename(os.path.abspath(path)))
    fps = int(data.get("fps", backend.default_fps))

    character = str(data.get("character", "")) or (primary.prompt if primary else "")
    ref = data.get("character_ref")
    character_ref = (os.path.join(path, str(ref)) if ref else "") or (primary.ref if primary else "")
    voice = primary.voice if primary else ""

    return Project(
        root=path, name=name, model=backend.name, fps=fps, shots=shots, engines=engines,
        cast_map=cast_map, cast_elements=cast_elements,
        character=character, character_ref=character_ref, cast=primary_role and cast_map[primary_role], voice=voice,
        scene=scene, scene_ref=(scene_el.ref if scene_el else ""), scene_desc=(scene_el.prompt if scene_el else ""), scene_el=scene_el,
        style=style, style_positive=(style_el.prompt if style_el else ""),
        style_negative=(style_el.negative if style_el else ""), style_video_suffix=style_video_suffix, style_el=style_el,
    )
