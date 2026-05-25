"""The composer — turns a (project, shot) into an engine-ready spec by merging the
*contributions* of the elements the shot uses. This is the role-aware glue:
  - subjects (cast roles in frame)  -> their appearance + reference image + LoRA
  - scene                            -> reference image (location/style/aspect)
  - style                            -> positive fragment + negative + LoRA
  - base (chaining)                  -> another shot's frame as the primary reference
  - speaker                          -> the TTS voice
Glue instruction templates live in compose.yaml (editable), never in engine code.
"""
from __future__ import annotations

import os

from .engines.base import ImageSpec, Lora, MotionSpec, VoiceSpec
from .manifest import Project, Shot

_GLUE_CACHE: dict | None = None


def _glue() -> dict:
    global _GLUE_CACHE
    if _GLUE_CACHE is None:
        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compose.yaml")
        _GLUE_CACHE = (yaml.safe_load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}) or {}
    return _GLUE_CACHE


def _require(path: str, what: str) -> str:
    if not path or not os.path.isfile(path):
        raise SystemExit(f"[compose] {what} 不存在：{path or '(未设置)'} —— 先生成它再出帧。")
    return path


def build_image_spec(project: Project, shot: Shot) -> ImageSpec:
    g = _glue()
    parts: list[str] = []
    refs: list[str] = []
    loras: list[Lora] = []

    frames_by_id = {s.id: s.frame for s in project.shots}

    if shot.base:
        base_frame = frames_by_id.get(shot.base)
        refs.append(_require(base_frame, f"{shot.id} 基于的 {shot.base} 帧"))
        # subjects' face refs reinforce identity (base frame already carries seat/scene)
        for role in shot.subjects:
            el = project.cast_elements.get(role)
            if el and el.ref:
                refs.append(_require(el.ref, f"角色 {role} 的参考图"))
                if el.lora:
                    loras.append(Lora(os.path.basename(el.lora), el.lora_trigger))
        parts += [shot.frame_prompt, project.style_positive]
        glue = g.get("base_reframe" if shot.base_mode == "reframe" else "base", "")
    else:
        for role in shot.subjects:                       # cast members in frame
            el = project.cast_elements.get(role)
            if not el:
                raise SystemExit(f"[compose] {shot.id} 的 subject '{role}' 不在 cast 里")
            if el.prompt:
                parts.append(el.prompt)
            if el.ref:
                refs.append(_require(el.ref, f"角色 {role} 的参考图"))
            if el.lora:
                loras.append(Lora(os.path.basename(el.lora), el.lora_trigger))
        if project.scene_ref:                            # the set
            refs.append(_require(project.scene_ref, "场景参考图"))
        parts += [shot.frame_prompt, project.style_positive]
        if project.style_el and project.style_el.lora:   # style LoRA (a modifier block)
            loras.append(Lora(os.path.basename(project.style_el.lora), project.style_el.lora_trigger))
        if shot.subjects and project.scene_ref:
            glue = g.get("char_scene", "")
        elif shot.subjects:
            glue = g.get("char_only", "")
        elif project.scene_ref:
            glue = g.get("scene_only", "")
        else:
            glue = ""

    prompt = "，".join(p for p in parts if p and p.strip())
    prompt = f"{prompt}。{glue}{g.get('aspect', '')}"
    return ImageSpec(
        out=shot.frame, prompt=prompt, refs=refs, loras=loras[:3],
        negative=project.style_negative, width=shot.width, height=shot.height, seed=shot.seed,
    )


def build_end_image_spec(project: Project, shot: Shot) -> ImageSpec:
    """Generate the END keyframe as an EDIT of the start frame — same scene/view/
    light, motion advanced a little — so start↔end stay consistent for FLF2V."""
    g = _glue()
    refs = [_require(shot.frame, f"{shot.id} 的起始帧（生成尾帧需要它）")]
    loras: list[Lora] = []
    if project.style_el and project.style_el.lora:
        loras.append(Lora(os.path.basename(project.style_el.lora), project.style_el.lora_trigger))
    parts = [shot.end_content.strip(), project.style_positive]
    prompt = "，".join(p for p in parts if p and p.strip())
    prompt = f"{prompt}。{g.get('end', '')}{g.get('aspect', '')}"
    return ImageSpec(out=shot.end_frame, prompt=prompt, refs=refs, loras=loras[:3],
                     negative=project.style_negative, width=shot.width, height=shot.height, seed=shot.seed + 7)


def build_motion_spec(project: Project, shot: Shot, out: str, t2v: bool = False) -> MotionSpec:
    # I2V: the frame carries appearance/scene, so the prompt is just camera+action+suffix.
    # T2V: no frame — prepend the subjects' appearance + scene description so the text
    # alone describes the whole shot (repeated verbatim each clip for consistency).
    prompt = shot.prompt
    if t2v:
        parts = [project.cast_elements[r].prompt for r in shot.subjects if project.cast_elements.get(r)]
        if project.scene_desc:
            parts.append(project.scene_desc)
        parts.append(shot.prompt)
        prompt = "，".join(p for p in parts if p and p.strip())
    return MotionSpec(
        frame=shot.frame, out=out, prompt=prompt, negative=shot.negative,
        seconds=shot.seconds, fps=project.fps, seed=shot.seed,
        width=shot.width, height=shot.height, end_frame=shot.end_frame,
    )


def build_voice_spec(project: Project, shot: Shot, out: str) -> VoiceSpec:
    return VoiceSpec(text=shot.dialogue, out=out, voice=project.voice_for(shot.speaker))
