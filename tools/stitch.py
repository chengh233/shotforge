"""Concatenate a project's per-shot mp4s into one continuous film.

    python -m tools.stitch --project projects/example

Tries a fast stream copy first; if the clips' codecs/params differ and the
concat demuxer refuses, falls back to a re-encode.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

from shotforge.manifest import load_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch a project's shot mp4s into one file.")
    parser.add_argument("--project", required=True, help="dir containing project.yaml")
    parser.add_argument("--out", default=None, help="output mp4 (default <project>/out/<name>_full.mp4)")
    args = parser.parse_args()

    project = load_project(args.project)
    outdir = os.path.join(args.project, "out")

    clips: list[str] = []
    for shot in project.shots:  # manifest order
        clip = os.path.join(outdir, f"{shot.id}.mp4")
        if os.path.isfile(clip):
            clips.append(clip)
        else:
            print(f"[skip] missing clip: {clip}")

    if not clips:
        print("[error] no clips to stitch; render the project first.")
        return

    out = args.out or os.path.join(outdir, f"{project.name}_full.mp4")

    # The concat demuxer reads a list file of `file '<path>'` lines.
    fd, list_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for clip in clips:
                fh.write(f"file '{os.path.abspath(clip)}'\n")

        base = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path]
        print(f"[stitch] {len(clips)} clips -> {out} (stream copy)")
        try:
            subprocess.run(base + ["-c", "copy", out], check=True)
        except subprocess.CalledProcessError:
            print("[stitch] stream copy failed; re-encoding")
            subprocess.run(base + ["-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)
        print(f"[ok] {out}")
    finally:
        os.unlink(list_path)


if __name__ == "__main__":
    main()
