"""Post (后期): assemble the final film — concatenate (or crossfade) the silent
clips, lay each shot's voiceover on a matching timeline, optionally mix in
background music, burn in subtitles, and optionally fade in/out.

    python -m tools.post --project projects/lasttram
    python -m tools.post --project projects/lasttram --crossfade 0.5 --fade 0.6
    python -m tools.post --project projects/lasttram --music bgm.mp3

Inputs (from earlier stages):
  <project>/out/<id>.mp4        per-shot silent clips    (shotforge generate)
  <project>/out/audio/<id>.mp3  per-shot voiceover       (tools.dub)       optional
Subtitles are built HERE from each shot's `dialogue`, timed to the final
timeline, so they stay aligned even with --crossfade (no need to run subs first).
Output: <project>/out/<name>_final.mp4

--crossfade S : dissolve S seconds between shots (default 0 = hard cut).
--fade F      : fade in/out F seconds at the very start/end (default 0).
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import tempfile

from shotforge.manifest import load_project

AR = "44100"  # uniform audio rate so segments concat/crossfade cleanly
# A font with CJK glyphs, or Chinese subtitles burn in as boxes (tofu). macOS
# ships PingFang; on Linux/Colab install fonts-noto-cjk (colab_setup.py does).
DEFAULT_FONT = "PingFang SC" if platform.system() == "Darwin" else "Noto Sans CJK SC"


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("[ffmpeg]", " ".join(cmd[:8]), "…")
    subprocess.run(cmd, check=True, cwd=cwd)


def _dur(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _ts(t: float) -> str:
    t = max(0.0, t)
    h, m, s, ms = int(t // 3600), int((t % 3600) // 60), int(t % 60), int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the final film (concat/crossfade + VO + music + subs).")
    ap.add_argument("--project", required=True)
    ap.add_argument("--music", default=None, help="background-music file (optional)")
    ap.add_argument("--music-volume", type=float, default=0.25)
    ap.add_argument("--crossfade", type=float, default=0.0, help="dissolve seconds between shots")
    ap.add_argument("--fade", type=float, default=0.0, help="fade in/out seconds at start/end")
    ap.add_argument("--font", default=DEFAULT_FONT, help="subtitle font (must have CJK glyphs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    project = load_project(args.project)
    outdir = os.path.join(args.project, "out")
    adir = os.path.join(outdir, "audio")
    final = args.out or os.path.join(outdir, f"{project.name}_final.mp4")

    def _clip(s):   # prefer the lip-synced clip when lipsync has run
        synced = os.path.join(outdir, f"{s.id}.synced.mp4")
        return synced if os.path.isfile(synced) else os.path.join(outdir, f"{s.id}.mp4")
    clips = [(s, _clip(s)) for s in project.shots if os.path.isfile(_clip(s))]
    if not clips:
        print("[error] no clips in out/; render first")
        return

    work = tempfile.mkdtemp()
    try:
        durs = [_dur(c) for _, c in clips]
        n = len(clips)
        D = max(0.0, args.crossfade)
        D = min(D, min(durs) * 0.5) if (D > 0 and n >= 2) else 0.0
        # start time of each shot on the final timeline (crossfade pulls each shot
        # D earlier than the previous one's end).
        starts = [sum(durs[:i]) - i * D for i in range(n)]

        # 1) per-shot audio aligned to each clip's duration (VO padded, else silence)
        segs = []
        for i, (shot, _) in enumerate(clips):
            seg = os.path.join(work, f"a{i:03d}.wav")
            vo = os.path.join(adir, f"{shot.id}.mp3")
            if os.path.isfile(vo):
                _run(["ffmpeg", "-y", "-i", vo, "-af", "apad", "-t", f"{durs[i]}", "-ar", AR, "-ac", "2", seg])
            else:
                _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={AR}:cl=stereo", "-t", f"{durs[i]}", seg])
            segs.append(seg)

        silent = os.path.join(work, "silent.mp4")
        vo_track = os.path.join(work, "vo.wav")

        if D > 0:
            # video: chain xfade dissolves; audio: chain acrossfade (same length)
            vin: list[str] = []
            for _, c in clips:
                vin += ["-i", os.path.abspath(c)]
            vfc, prev, L = [], "[0:v]", durs[0]
            for i in range(1, n):
                vfc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={D:.3f}:offset={L - D:.3f}[v{i}]")
                prev, L = f"[v{i}]", L + durs[i] - D
            _run(["ffmpeg", "-y", *vin, "-filter_complex", ";".join(vfc),
                  "-map", prev, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", silent])

            ain: list[str] = []
            for s in segs:
                ain += ["-i", s]
            afc, prev = [], "[0:a]"
            for i in range(1, n):
                afc.append(f"{prev}[{i}:a]acrossfade=d={D:.3f}[a{i}]")
                prev = f"[a{i}]"
            _run(["ffmpeg", "-y", *ain, "-filter_complex", ";".join(afc), "-map", prev, vo_track])
        else:
            # hard cut: concat (re-encode video for a safe concat)
            vlist = os.path.join(work, "vid.txt")
            with open(vlist, "w") as fh:
                for _, c in clips:
                    fh.write(f"file '{os.path.abspath(c)}'\n")
            _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", vlist,
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", silent])
            alist = os.path.join(work, "vo.txt")
            with open(alist, "w") as fh:
                for s in segs:
                    fh.write(f"file '{s}'\n")
            _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", alist, "-c", "copy", vo_track])

        total = _dur(silent)

        # 2) final audio = voiceover (+ optional ducked, looped music)
        if args.music and os.path.isfile(args.music):
            final_audio = os.path.join(work, "mix.wav")
            _run(["ffmpeg", "-y", "-i", vo_track, "-stream_loop", "-1", "-i", args.music,
                  "-filter_complex",
                  f"[1:a]volume={args.music_volume},atrim=0:{total},aresample={AR}[m];"
                  f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
                  "-map", "[a]", final_audio])
        else:
            final_audio = vo_track

        # 3) subtitles from each shot's dialogue, timed to the final timeline
        cues = []
        for i, (shot, _) in enumerate(clips):
            if not shot.dialogue.strip():
                continue
            end = starts[i + 1] if i + 1 < n else total
            cues.append(f"{len(cues) + 1}\n{_ts(starts[i])} --> {_ts(end)}\n{shot.dialogue.strip()}\n")
        with open(os.path.join(work, "subs.srt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(cues))

        # 4) mux: burn subtitles + optional fades
        vf = []
        if cues:
            vf.append(f"subtitles=subs.srt:force_style='Fontname={args.font}'")
        if args.fade > 0:
            vf.append(f"fade=t=in:st=0:d={args.fade:.3f}")
            vf.append(f"fade=t=out:st={max(0.0, total - args.fade):.3f}:d={args.fade:.3f}")
        cmd = ["ffmpeg", "-y", "-i", silent, "-i", final_audio]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if args.fade > 0:
            cmd += ["-af", f"afade=t=in:d={args.fade:.3f},"
                            f"afade=t=out:st={max(0.0, total - args.fade):.3f}:d={args.fade:.3f}"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", os.path.abspath(final)]
        print(f"[post] mux: {n} shots | crossfade={D:.2f}s | fade={args.fade:.2f}s | font={args.font}")
        subprocess.run(cmd, check=True, cwd=work)
        print(f"[ok] -> {final}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
