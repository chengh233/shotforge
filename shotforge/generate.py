"""CLI: render every shot in a project to an mp4.

    python -m shotforge.generate --project projects/example
    python -m shotforge.generate --project projects/example --shot s2

The project's ``model:`` field selects the backend; per-shot frame counts and
dimensions follow that backend's constraints (see shotforge/backends.py).
"""
from __future__ import annotations

import argparse
import os

from diffusers.utils import export_to_video

from . import i2v
from .backends import get_backend
from .manifest import frames_for, load_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Render shotforge project shots to mp4.")
    parser.add_argument("--project", required=True, help="dir containing project.yaml")
    parser.add_argument("--shot", default=None, help="render only this shot id")
    parser.add_argument("--outdir", default=None, help="output dir (default <project>/out)")
    args = parser.parse_args()

    project = load_project(args.project)
    backend = get_backend(project.model)
    outdir = args.outdir or os.path.join(args.project, "out")
    os.makedirs(outdir, exist_ok=True)

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


if __name__ == "__main__":
    main()
