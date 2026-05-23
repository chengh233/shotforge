"""CLI: render every shot in a project to an mp4.

    python -m shotforge.generate --project projects/example
    python -m shotforge.generate --project projects/example --shot s2
    python -m shotforge.generate --project projects/example --engine comfy

Engines:
  diffusers (default) — local diffusers pipeline (see i2v.py / backends.py).
  comfy               — drive a running ComfyUI server over HTTP (see comfy.py /
                        docs/COMFYUI.md). The high-quality Wan path; diffusers'
                        Wan I2V melts.
"""
from __future__ import annotations

import argparse
import os

from .backends import get_backend
from .manifest import frames_for, load_project


def _render_comfy(project, args, outdir) -> None:
    from . import comfy

    backend = get_backend(project.model)  # used only for the area budget / dim multiple
    workflow = comfy.load_workflow(args.workflow)
    print(f"[comfy] server={args.comfy_url} | workflow={args.workflow}")
    print(f"[project] {project.name} | engine=comfy | fps={project.fps} | shots={len(project.shots)}")

    for shot in project.shots:
        if args.shot and shot.id != args.shot:
            continue
        if not os.path.isfile(shot.frame):
            print(f"[skip] {shot.id}: frame not found -> {shot.frame}")
            continue

        width, height = comfy.derive_dims(shot.frame, shot.width * shot.height, backend.dim_multiple)
        out_path = os.path.join(outdir, f"{shot.id}.mp4")
        print(f"[render] {shot.id} | {width}x{height} | {shot.seconds}s @ {project.fps}fps | seed={shot.seed}")
        comfy.render_shot(
            workflow,
            args.comfy_url,
            frame_path=shot.frame,
            prompt=shot.prompt,
            negative=shot.negative,
            width=width,
            height=height,
            seconds=shot.seconds,
            fps=project.fps,
            seed=shot.seed,
            out_path=out_path,
        )
        print(f"[ok] {shot.id} -> {out_path}")


def _render_diffusers(project, args, outdir) -> None:
    from diffusers.utils import export_to_video

    from . import i2v

    backend = get_backend(project.model)
    device, dtype = i2v.device_dtype()
    print(f"[device] {device} ({dtype})")
    print(
        f"[project] {project.name} | model={backend.name} | "
        f"fps={project.fps} | shots={len(project.shots)}"
    )

    for shot in project.shots:
        if args.shot and shot.id != args.shot:
            continue
        if not os.path.isfile(shot.frame):
            print(f"[skip] {shot.id}: frame not found -> {shot.frame}")
            continue

        nf = frames_for(shot.seconds, project.fps, backend.frame_quantum)
        print(
            f"[render] {shot.id} | {shot.width}x{shot.height} | "
            f"{shot.seconds}s -> {nf} frames | steps={shot.steps} | seed={shot.seed}"
        )
        video = i2v.generate(
            backend=backend,
            frame_path=shot.frame,
            prompt=shot.prompt,
            negative=shot.negative,
            width=shot.width,
            height=shot.height,
            num_frames=nf,
            steps=shot.steps,
            seed=shot.seed,
        )
        out_path = os.path.join(outdir, f"{shot.id}.mp4")
        export_to_video(video, out_path, fps=project.fps)
        print(f"[ok] {shot.id} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render shotforge project shots to mp4.")
    parser.add_argument("--project", required=True, help="dir containing project.yaml")
    parser.add_argument("--shot", default=None, help="render only this shot id")
    parser.add_argument("--outdir", default=None, help="output dir (default <project>/out)")
    parser.add_argument(
        "--engine", choices=("diffusers", "comfy"), default="diffusers",
        help="diffusers (local pipeline) or comfy (drive a ComfyUI server)",
    )
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--workflow", default="comfyui/wan_i2v_api.json", help="ComfyUI API workflow JSON")
    args = parser.parse_args()

    project = load_project(args.project)
    outdir = args.outdir or os.path.join(args.project, "out")
    os.makedirs(outdir, exist_ok=True)

    if args.engine == "comfy":
        _render_comfy(project, args, outdir)
    else:
        _render_diffusers(project, args, outdir)


if __name__ == "__main__":
    main()
