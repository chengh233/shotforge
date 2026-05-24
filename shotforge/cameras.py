"""Camera-move vocabulary (运镜) — decouples HOW the camera moves from WHAT is in
the scene. A shot says ``camera: push_in``; the move's Chinese fragment is
prepended to the shot's action to build the Wan motion prompt (the style's
``video_suffix`` is appended after). Extend or override any id with a repo-level
``cameras.yaml`` (``id: 中文片段``).
"""
from __future__ import annotations

import os

import yaml

# Named camera moves -> Chinese motion-prompt fragment.
CAMERAS: dict[str, str] = {
    "push_in": "镜头缓缓推近",
    "pull_back": "镜头缓缓向后拉远",
    "pull_back_reveal": "镜头缓缓向后拉远，逐渐展开整个空间",
    "pull_back_tilt": "镜头缓缓向后拉远并轻微上摇，逐渐展开窗外的天空",
    "pan_left": "镜头向左缓缓横移，产生视差",
    "pan_right": "镜头向右缓缓横移，产生视差",
    "push_left": "镜头缓缓推近、同时轻轻移向左侧，聚焦画面左侧的人物",
    "push_right": "镜头缓缓推近、同时轻轻移向右侧，聚焦画面右侧的人物",
    "tilt_up": "镜头轻微上摇",
    "ots": "过肩镜头，越过人物肩头看向前方",
    "static": "镜头几乎静止，仅有轻微的呼吸般晃动",
    "handheld": "轻微手持晃动感",
}


def cameras() -> dict[str, str]:
    """Built-in moves merged with an optional repo-level ``cameras.yaml`` override."""
    merged = dict(CAMERAS)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cameras.yaml")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            extra = yaml.safe_load(fh) or {}
        merged.update({str(k): str(v) for k, v in extra.items()})
    return merged
