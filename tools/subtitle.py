"""Subtitles (字幕): build an SRT from each shot's `dialogue`, timed to the
rendered clip lengths.

    python -m tools.subtitle --project projects/example

Reads <project>/out/<id>.mp4 durations (ffprobe) in manifest order and writes
<project>/out/<name>.srt — each shot's dialogue spans its clip. tools.post burns
this in. (Shots with no dialogue produce no cue.)
"""
from __future__ import annotations

import argparse
import os
import subprocess

from shotforge.manifest import load_project


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an SRT timed to the rendered clips.")
    ap.add_argument("--project", required=True)
    args = ap.parse_args()

    project = load_project(args.project)
    outdir = os.path.join(args.project, "out")

    t = 0.0
    cues: list[str] = []
    for shot in project.shots:
        clip = os.path.join(outdir, f"{shot.id}.mp4")
        if not os.path.isfile(clip):
            print(f"[skip] {shot.id}: no clip")
            continue
        start, t = t, t + _duration(clip)
        if shot.dialogue.strip():
            cues.append(f"{len(cues) + 1}\n{_ts(start)} --> {_ts(t)}\n{shot.dialogue.strip()}\n")

    srt = os.path.join(outdir, f"{project.name}.srt")
    with open(srt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(cues))
    print(f"[ok] {len(cues)} cues -> {srt}")


if __name__ == "__main__":
    main()
