"""Voiceover (配音): TTS each shot's `dialogue` to an audio clip.

    python -m tools.dub --project projects/example
    python -m tools.dub --project projects/example --voice zh-CN-YunxiNeural

Uses edge-tts (Microsoft Edge's online TTS) — no GPU, good Chinese voices, free.
Writes <project>/out/audio/<shot id>.mp3 for every shot that has a `dialogue`.

Higher-quality / voice-cloning alternatives (local, GPU) — see docs/STAGES.md:
  CosyVoice2 (Alibaba), IndexTTS2, F5-TTS.

Some zh-CN voices: zh-CN-XiaoxiaoNeural (f), zh-CN-XiaoyiNeural (f),
zh-CN-YunxiNeural (m), zh-CN-YunjianNeural (m).  `edge-tts --list-voices` for all.
"""
from __future__ import annotations

import argparse
import asyncio
import os

import edge_tts

from shotforge.manifest import load_project


async def _say(text: str, voice: str, path: str) -> None:
    await edge_tts.Communicate(text, voice).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="TTS each shot's dialogue (edge-tts).")
    ap.add_argument("--project", required=True)
    ap.add_argument("--voice", default="zh-CN-XiaoxiaoNeural")
    args = ap.parse_args()

    project = load_project(args.project)
    adir = os.path.join(args.project, "out", "audio")
    os.makedirs(adir, exist_ok=True)

    n = 0
    for shot in project.shots:
        if not shot.dialogue.strip():
            print(f"[skip] {shot.id}: no dialogue")
            continue
        path = os.path.join(adir, f"{shot.id}.mp3")
        print(f"[tts] {shot.id}: {shot.dialogue!r} -> {path}")
        asyncio.run(_say(shot.dialogue, args.voice, path))
        n += 1
    print(f"[ok] {n} clips -> {adir}")


if __name__ == "__main__":
    main()
