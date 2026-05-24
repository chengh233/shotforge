"""Post (后期): assemble the final film — concat the silent clips, lay the
per-shot voiceover on a matching timeline, optionally mix in background music,
and burn in the subtitles.

    python -m tools.post --project projects/example
    python -m tools.post --project projects/example --music bgm.mp3 --music-volume 0.2

Inputs (from earlier stages):
  <project>/out/<id>.mp4        per-shot silent clips    (shotforge generate)
  <project>/out/audio/<id>.mp3  per-shot voiceover       (tools.dub)       optional
  <project>/out/<name>.srt      subtitles                (tools.subtitle)  optional
Output:
  <project>/out/<name>_final.mp4

Background music: generate one with an AI model (MusicGen / Stable Audio Open —
see docs/STAGES.md) or use a royalty-free track, then pass it via --music.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

from shotforge.manifest import load_project

AR = "44100"  # uniform audio rate so segments concat cleanly


def _run(cmd: list[str]) -> None:
    print("[ffmpeg]", " ".join(cmd[:8]), "…")
    subprocess.run(cmd, check=True)


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Mux clips + voiceover + music + subtitles into the final film.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--music", default=None, help="background-music file (optional)")
    ap.add_argument("--music-volume", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    project = load_project(args.project)
    outdir = os.path.join(args.project, "out")
    adir = os.path.join(outdir, "audio")
    final = args.out or os.path.join(outdir, f"{project.name}_final.mp4")

    clips = [(s, os.path.join(outdir, f"{s.id}.mp4")) for s in project.shots
             if os.path.isfile(os.path.join(outdir, f"{s.id}.mp4"))]
    if not clips:
        print("[error] no clips in out/; render first")
        return

    work = tempfile.mkdtemp()
    try:
        # 1) per-shot audio aligned to each clip's duration (VO padded to length,
        #    or silence when the shot has no voiceover).
        segs = []
        for i, (shot, clip) in enumerate(clips):
            dur = _duration(clip)
            seg = os.path.join(work, f"a{i:03d}.wav")
            vo = os.path.join(adir, f"{shot.id}.mp3")
            if os.path.isfile(vo):
                _run(["ffmpeg", "-y", "-i", vo, "-af", "apad", "-t", f"{dur}", "-ar", AR, "-ac", "2", seg])
            else:
                _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={AR}:cl=stereo", "-t", f"{dur}", seg])
            segs.append(seg)
        vo_list = os.path.join(work, "vo.txt")
        with open(vo_list, "w") as fh:
            for p in segs:
                fh.write(f"file '{p}'\n")
        vo_track = os.path.join(work, "vo.wav")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", vo_list, "-c", "copy", vo_track])

        # 2) concat the silent video clips (re-encode for a safe concat).
        vid_list = os.path.join(work, "vid.txt")
        with open(vid_list, "w") as fh:
            for _, c in clips:
                fh.write(f"file '{os.path.abspath(c)}'\n")
        silent = os.path.join(work, "silent.mp4")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", vid_list,
              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", silent])
        total = _duration(silent)

        # 3) final audio = voiceover (+ optional ducked, looped music).
        if args.music and os.path.isfile(args.music):
            final_audio = os.path.join(work, "mix.wav")
            _run(["ffmpeg", "-y", "-i", vo_track, "-stream_loop", "-1", "-i", args.music,
                  "-filter_complex",
                  f"[1:a]volume={args.music_volume},atrim=0:{total},aresample={AR}[m];"
                  f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
                  "-map", "[a]", final_audio])
        else:
            final_audio = vo_track

        # 4) mux audio + burn subtitles (copy srt to an ascii path for the filter).
        vf = []
        srt = os.path.join(outdir, f"{project.name}.srt")
        if os.path.isfile(srt):
            srt_local = os.path.join(work, "subs.srt")
            shutil.copyfile(srt, srt_local)
            vf = ["-vf", "subtitles=subs.srt"]
        cmd = ["ffmpeg", "-y", "-i", silent, "-i", final_audio, *vf,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", os.path.abspath(final)]
        # run from work/ so the relative subtitles path resolves
        print("[ffmpeg] mux + burn subtitles …")
        subprocess.run(cmd, check=True, cwd=work)
        print(f"[ok] -> {final}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
