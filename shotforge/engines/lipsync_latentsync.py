"""Lip-sync engine (P4): re-time a talking character's mouth to its voice audio.

Open-source models (run on the GPU): LatentSync (ByteDance), MuseTalk, SadTalker,
Wav2Lip. shotforge calls one via a configurable CLI or ComfyUI workflow; set it
with env. Applied only to shots where the speaker is on-screen and has dialogue.

⚠️ EXPERIMENTAL for ANIME faces — most lip-sync models are trained on real faces,
so anime mouth-sync is hit-or-miss. The base pipeline works fine WITHOUT lip-sync
(voiceover over loose mouth motion, the usual anime convention); turn this on only
if you want tighter sync.

Config:
  $LIPSYNC_CMD   a shell template with {video} {audio} {out}, e.g.
                 "python /content/LatentSync/inference.py --video {video} --audio {audio} --out {out}"
"""
from __future__ import annotations

import os
import shlex
import subprocess

LIPSYNC_CMD = os.environ.get("LIPSYNC_CMD", "")


class LatentSyncEngine:
    name = "latentsync"

    def sync(self, video: str, audio: str, out: str) -> None:
        if not LIPSYNC_CMD:
            raise SystemExit(
                "[lipsync] 未配置 lip-sync 命令。设 $LIPSYNC_CMD（含 {video}{audio}{out}）指向 "
                "LatentSync/MuseTalk 等的推理脚本。\n  动漫脸效果有限——不开口型同步也能出片（旁白式）。"
            )
        cmd = LIPSYNC_CMD.format(video=shlex.quote(video), audio=shlex.quote(audio), out=shlex.quote(out))
        print(f"[lipsync] $ {cmd}")
        subprocess.run(cmd, shell=True, check=True)


ENGINE = LatentSyncEngine()
