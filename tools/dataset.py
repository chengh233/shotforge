"""Build a character LoRA training set from ONE base image: edit it into ~20 varied
views (same person, different angle/expression, plain background) via an image-EDIT
engine (default Qwen-Image-Edit). Plain backgrounds so the LoRA learns the PERSON,
not scenes. Optionally writes a trigger-word caption beside each image.

    python -m tools.dataset --base characters/mira/base.png --out characters/mira/dataset
    python -m tools.dataset --base base.png --out ds --trigger mira --n 2

Needs the qwen image engine set up (scripts/qwen_setup.py + comfyui/qwen_edit_api.json).
"""
from __future__ import annotations

import argparse
import os

# ~20 varied views/expressions on a plain background (edit instructions for the base).
DEFAULT_VIEWS = [
    "the same person, front-facing head-and-shoulders portrait, neutral calm expression",
    "the same person, three-quarter view turned slightly left",
    "the same person, three-quarter view turned slightly right",
    "the same person, left side profile",
    "the same person, right side profile",
    "the same person, looking slightly upward, gentle soft smile",
    "the same person, looking slightly downward, calm",
    "the same person, tight close-up of the face, warm smile",
    "the same person, faintly sad expression, eyes a little downcast",
    "the same person, slightly surprised expression, eyes a bit wide",
    "the same person, laughing, eyes happily narrowed",
    "the same person, three-quarter back view glancing over the shoulder",
    "the same person, upper body, arms relaxed at the sides, standing straight",
    "the same person, head tilted slightly, thoughtful look",
    "the same person, soft side lighting on the face, neutral expression",
    "the same person, bright even front lighting, gentle smile",
    "the same person, looking directly at the camera, confident",
    "the same person, chin slightly down, eyes up to camera",
    "the same person, hair gently moved as if by a light breeze, calm",
    "the same person, three-quarter view, warm friendly smile",
]
_SUFFIX = ("，保持与参考图完全相同的人物身份（脸、发型、五官），纯净浅灰背景，"
           "写实人像，高清细腻。Keep the exact same identity as the reference image; plain light-gray background.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a character LoRA dataset from one base image (image-edit).")
    ap.add_argument("--base", required=True, help="the single base image of the character")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--prompts", default=None, help="text file of edit prompts (one per line); default: built-in views")
    ap.add_argument("--engine", default="qwen", help="image engine (must support refs/edit)")
    ap.add_argument("--n", type=int, default=1, help="images per prompt")
    ap.add_argument("--trigger", default=None, help="if set, write a <trigger> caption .txt beside each image")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(a.base):
        raise SystemExit(f"[dataset] base image not found: {a.base}")
    from shotforge.engines.base import ImageSpec, get_engine

    if a.prompts:
        with open(a.prompts, encoding="utf-8") as fh:
            views = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        views = DEFAULT_VIEWS
    eng = get_engine("image", a.engine)
    os.makedirs(a.out, exist_ok=True)
    print(f"[dataset] base={a.base} | engine={a.engine} | {len(views)} views x{a.n} -> {a.out}")

    idx = 0
    for view in views:
        for _ in range(max(1, a.n)):
            idx += 1
            out = os.path.join(a.out, f"{idx:03d}.png")
            if os.path.isfile(out) and not a.overwrite:
                print(f"[skip] {out} exists")
                continue
            spec = ImageSpec(out=out, prompt=view + _SUFFIX, refs=[a.base],
                             width=a.width, height=a.height, seed=idx)
            print(f"[dataset] {idx:03d} {view[:40]}… -> {out}")
            eng.generate(spec)
            if a.trigger:
                with open(os.path.splitext(out)[0] + ".txt", "w", encoding="utf-8") as fh:
                    fh.write(a.trigger)
    print(f"[ok] dataset -> {a.out}  (curate: delete any off-identity images before training)")


if __name__ == "__main__":
    main()
