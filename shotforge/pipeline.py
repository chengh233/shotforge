"""Stage orchestration: read a project, compose each shot into an engine spec, run
the selected engine. Content-free — all prompts/refs/loras come from the libraries
via the composer; engines come from project.yaml ``engines:`` (overridable by --engine).

    python -m shotforge.pipeline frames  --project projects/cafe [--engine ..] [--shot s1] [--variations 4] [--overwrite]
    python -m shotforge.pipeline video   --project projects/cafe
    python -m shotforge.pipeline lipsync --project projects/cafe
"""
from __future__ import annotations

import argparse
import os

from . import compose
from .engines.base import get_engine
from .manifest import load_project


def run_frames(project_dir, engine=None, shot=None, variations=1, overwrite=False):
    project = load_project(project_dir)
    if project.engines.get("image_model"):   # e.g. gemini-3-pro-image-preview (Nano Banana Pro)
        os.environ["NANOBANANA_MODEL"] = project.engines["image_model"]
    name = engine or project.engines.get("image", "qwen")   # 默认 Qwen-Image（全开源）
    eng = get_engine("image", name)
    print(f"[frames] {project.name} | engine={name} | cast={project.cast_map} | scene={project.scene} | style={project.style}")
    for s in project.shots:
        if shot and s.id != shot:
            continue
        if not s.frame_prompt.strip():
            print(f"[skip] {s.id}: no content/frame_prompt")
            continue
        # a still master is the spatial anchor for `base:` chaining — always one
        # file (s0.jpeg), never variation-suffixed, so base lookups resolve.
        vcount = 1 if s.still else max(1, variations)
        base, ext = os.path.splitext(s.frame)
        for v in range(vcount):
            out = s.frame if vcount <= 1 else f"{base}_{v + 1}{ext}"
            if os.path.isfile(out) and not overwrite:
                print(f"[skip] {out} exists (use --overwrite)")
                continue
            spec = compose.build_image_spec(project, s)
            spec.out = out
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tag = "" if variations <= 1 else f" v{v + 1}"
            print(f"[frames] {s.id}{tag} | subjects={s.subjects or '—'} | refs={len(spec.refs)} | loras={len(spec.loras)} -> {out}")
            eng.generate(spec)
        # FLF2V only: generate the END keyframe (edit of the start). With a regular
        # I2V video engine the end isn't needed, so skip it (no Qwen-Edit setup required).
        if s.end_content and "flf" in str(project.engines.get("video", "")) and os.path.isfile(s.frame):
            if os.path.isfile(s.end_frame) and not overwrite:
                print(f"[skip] {s.end_frame} exists (use --overwrite)")
            else:
                espec = compose.build_end_image_spec(project, s)
                print(f"[frames] {s.id} END (edit of start) -> {s.end_frame}")
                eng.generate(espec)
    print("[ok] frames done")


def run_video(project_dir, engine=None, shot=None):
    project = load_project(project_dir)
    name = engine or project.engines.get("video", "wan")
    eng = get_engine("video", name)
    t2v = "t2v" in name                      # text-to-video: no starting frame needed
    outdir = os.path.join(project_dir, "out")
    os.makedirs(outdir, exist_ok=True)
    print(f"[video] {project.name} | engine={name} | {'T2V' if t2v else 'I2V'} | fps={project.fps} | shots={len(project.shots)}")
    for s in project.shots:
        if shot and s.id != shot:
            continue
        if s.still:
            print(f"[skip] {s.id}: still master (not animated)")
            continue
        if not t2v and not os.path.isfile(s.frame):
            print(f"[skip] {s.id}: frame not found -> {s.frame}")
            continue
        out = os.path.join(outdir, f"{s.id}.mp4")
        print(f"[render] {s.id} | {s.seconds}s @ {project.fps}fps | seed={s.seed}")
        eng.animate(compose.build_motion_spec(project, s, out, t2v=t2v))
        print(f"[ok] {s.id} -> {out}")
    print("[ok] video done")


def run_lipsync(project_dir, engine=None, shot=None):
    """Sync speaking, on-screen shots to their voice audio (out/audio/<id>.mp3)."""
    project = load_project(project_dir)
    name = engine or project.engines.get("lipsync", "latentsync")
    eng = get_engine("lipsync", name)
    outdir = os.path.join(project_dir, "out")
    n = 0
    for s in project.shots:
        if shot and s.id != shot:
            continue
        speaking_on_screen = s.dialogue.strip() and s.speaker in s.subjects
        clip = os.path.join(outdir, f"{s.id}.mp4")
        audio = os.path.join(outdir, "audio", f"{s.id}.mp3")
        if not speaking_on_screen:
            print(f"[skip] {s.id}: speaker not on screen / no dialogue (voiceover only)")
            continue
        if not (os.path.isfile(clip) and os.path.isfile(audio)):
            print(f"[skip] {s.id}: need {clip} + {audio} (run video & voice first)")
            continue
        out = os.path.join(outdir, f"{s.id}.synced.mp4")
        print(f"[lipsync] {s.id}: {clip} + {audio} -> {out}")
        eng.sync(clip, audio, out)
        n += 1
    print(f"[ok] lipsync done ({n} shots)")


def main() -> None:
    ap = argparse.ArgumentParser(description="shotforge stage runner (compose + engines).")
    ap.add_argument("stage", choices=("frames", "video", "lipsync"))
    ap.add_argument("--project", required=True)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--shot", default=None)
    ap.add_argument("--variations", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.stage == "frames":
        run_frames(a.project, a.engine, a.shot, a.variations, a.overwrite)
    elif a.stage == "video":
        run_video(a.project, a.engine, a.shot)
    else:
        run_lipsync(a.project, a.engine, a.shot)


if __name__ == "__main__":
    main()
