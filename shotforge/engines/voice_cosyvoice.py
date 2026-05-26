"""Voice engine: CosyVoice (natural Chinese TTS, preset speaker '中文女', no reference).

Replaces edge-tts' synthetic feel. CosyVoice runs in its own venv (scripts/cosyvoice_setup.py),
so this engine shells out to that venv's python + tools/_cosyvoice_infer.py, then converts the
wav to the .mp3 the rest of the pipeline expects. Pick it per project with `engines: voice: cosyvoice`.
$COSYVOICE_SPK overrides the preset speaker.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

from .base import VoiceSpec

CV = os.environ.get("COSYVOICE", "/content/CosyVoice")
VENV_PY = os.environ.get("COSYVOICE_PY", "/content/cosyvoice-venv/bin/python")
SPK = os.environ.get("COSYVOICE_SPK", "中文女")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CosyVoiceEngine:
    name = "cosyvoice"

    def say(self, spec: VoiceSpec) -> None:
        if not os.path.isfile(VENV_PY):
            raise SystemExit("[cosyvoice] 未安装：先 python scripts/cosyvoice_setup.py")
        infer = os.path.join(_REPO, "tools", "_cosyvoice_infer.py")
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run([VENV_PY, infer, "--text", spec.text, "--out", wav, "--spk", SPK],
                           check=True, env=dict(os.environ, COSYVOICE=CV))
            os.makedirs(os.path.dirname(spec.out) or ".", exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, spec.out], check=True)
        finally:
            if os.path.isfile(wav):
                os.remove(wav)


ENGINE = CosyVoiceEngine()
