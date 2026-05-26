"""Make a still-image "video" the length of each shot's voiceover — a talking-head
clip to feed lip-sync when you don't want Wan motion (and aren't bound by Wan's ~5s
limit). Holds frames/<id>.jpeg for the duration of out/audio/<id>.mp3 at the project
fps, writing a silent out/<id>.mp4.

    python -m tools.still_clip --project projects/wanwan_intro
    python run.py stillclip projects/wanwan_intro

Run AFTER voiceover (tools.dub / `run.py dub`) so the audio exists to size the clip;
then `run.py lipsync` re-times the mouth, and `run.py post` assembles the final.
"""
from __future__ import annotations

import argparse
import os
import subprocess

from shotforge.manifest import load_project


def _dur(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Hold each shot's frame for its voiceover length -> out/<id>.mp4")
    ap.add_argument("--project", required=True)
    ap.add_argument("--shot", default=None)
    ap.add_argument("--pad", type=float, default=0.15, help="extra seconds held after the audio ends")
    a = ap.parse_args()

    project = load_project(a.project)
    outdir = os.path.join(a.project, "out")
    adir = os.path.join(outdir, "audio")
    os.makedirs(outdir, exist_ok=True)

    n = 0
    for s in project.shots:
        if a.shot and s.id != a.shot:
            continue
        if not os.path.isfile(s.frame):
            print(f"[skip] {s.id}: no frame {s.frame}")
            continue
        audio = os.path.join(adir, f"{s.id}.mp3")
        if not os.path.isfile(audio):
            print(f"[skip] {s.id}: no audio {audio} (run dub first)")
            continue
        dur = _dur(audio) + a.pad
        out = os.path.join(outdir, f"{s.id}.mp4")
        print(f"[still] {s.id}: hold {os.path.basename(s.frame)} for {dur:.2f}s @ {project.fps}fps -> {out}")
        subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-i", s.frame, "-t", f"{dur:.3f}",
             "-r", str(project.fps), "-pix_fmt", "yuv420p",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-an", out],
            check=True,
        )
        n += 1
    print(f"[ok] {n} still clip(s) -> {outdir}")


if __name__ == "__main__":
    main()
