"""Synthesize one line with CosyVoice-300M-SFT (preset speaker, no reference). Runs INSIDE
the cosyvoice venv (invoked by shotforge.engines.voice_cosyvoice), writes a wav.

    <cosyvoice-venv>/bin/python tools/_cosyvoice_infer.py --text "你好" --out a.wav [--spk 中文女]
"""
import argparse
import os
import sys

CV = os.environ.get("COSYVOICE", "/content/CosyVoice")
sys.path.insert(0, CV)
sys.path.insert(0, os.path.join(CV, "third_party", "Matcha-TTS"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spk", default="中文女")
    ap.add_argument("--model", default=os.path.join(CV, "pretrained_models", "CosyVoice-300M-SFT"))
    a = ap.parse_args()

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice

    m = CosyVoice(a.model)
    chunks = [r["tts_speech"] for r in m.inference_sft(a.text, a.spk, stream=False)]
    speech = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    torchaudio.save(a.out, speech, m.sample_rate)
    print(f"[cosyvoice] saved {a.out} ({m.sample_rate} Hz)")


if __name__ == "__main__":
    main()
