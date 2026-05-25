"""Engine interfaces + specs + a lazy registry.

Specs are the content-carrying values the composer hands to a content-free engine:
  ImageSpec  -> ImageEngine.generate   (frames stage)
  MotionSpec -> VideoEngine.animate     (video / I2V stage)
  VoiceSpec  -> VoiceEngine.say          (voice / TTS stage)
  (LipSyncEngine.sync takes a video + audio path; MusicEngine.score a prompt+dur.)

Selection is by name: ``get_engine("image", "nanobanana")`` imports
``shotforge.engines.image_nanobanana`` and returns its ``ENGINE``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Lora:
    path: str                 # file in ComfyUI/models/loras (or absolute)
    trigger: str = ""
    weight: float = 0.85


@dataclass
class ImageSpec:
    out: str                  # where to write the frame
    prompt: str               # composed positive prompt
    refs: list[str] = field(default_factory=list)    # reference image paths (subject, scene, base…)
    loras: list[Lora] = field(default_factory=list)  # ≤2-3 (interference)
    negative: str = ""
    width: int = 832
    height: int = 1216
    seed: int = 0


@dataclass
class MotionSpec:
    frame: str                # the starting frame (the manga panel)
    out: str
    prompt: str               # composed motion prompt (camera + action + style suffix)
    negative: str = ""
    seconds: float = 5.0
    fps: int = 16
    seed: int = 0
    width: int = 0            # 0 -> let the engine derive from the frame aspect
    height: int = 0
    end_frame: str = ""       # optional END keyframe (FLF2V engines fill the motion frame->end_frame)


@dataclass
class VoiceSpec:
    text: str
    out: str
    voice: str = ""           # TTS voice id, or a reference-audio path for cloning engines


@runtime_checkable
class ImageEngine(Protocol):
    name: str
    def generate(self, spec: ImageSpec) -> None: ...


@runtime_checkable
class VideoEngine(Protocol):
    name: str
    def animate(self, spec: MotionSpec) -> None: ...


@runtime_checkable
class VoiceEngine(Protocol):
    name: str
    def say(self, spec: VoiceSpec) -> None: ...


@runtime_checkable
class LipSyncEngine(Protocol):
    name: str
    def sync(self, video: str, audio: str, out: str) -> None: ...


# stage -> default module name prefix
_PREFIX = {"image": "image", "video": "video", "voice": "voice", "lipsync": "lipsync", "music": "music"}


def get_engine(stage: str, name: str):
    """Import ``shotforge.engines.<stage>_<name>`` and return its module-level ENGINE."""
    prefix = _PREFIX.get(stage, stage)
    mod_name = f"shotforge.engines.{prefix}_{name.replace('-', '_')}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"[engines] no {stage} engine '{name}' ({mod_name}). "
                         f"Check project.yaml `engines: {stage}: …`.") from exc
    engine = getattr(mod, "ENGINE", None)
    if engine is None:
        raise SystemExit(f"[engines] {mod_name} defines no ENGINE")
    return engine
