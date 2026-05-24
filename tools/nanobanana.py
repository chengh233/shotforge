"""Batch image generation with Google "Nano Banana" (Gemini 2.5 Flash Image),
reference-conditioned for character consistency. Runs on your Mac (it's an API,
no GPU) — keep the Colab GPU for video.

Two modes:

1) DATASET — make many varied images of one character (e.g. for a LoRA training
   set). One prompt per line in a text file; an optional reference image keeps
   the same face/outfit:
       python -m tools.nanobanana --prompts characters/yuki/dataset_prompts.txt \
           --ref characters/yuki/ref.png --out characters/yuki/dataset

2) FRAMES — generate a project's shot starting-frames directly (text-to-image
   with a reference for consistency; each shot gets its own composition from its
   frame_prompt). Saves to each shot's `frame` path:
       python -m tools.nanobanana --project projects/lasttram

Setup (once, on the Mac):
    pip install google-genai pillow
    export GEMINI_API_KEY=...        # from https://aistudio.google.com/apikey

The model id defaults to `gemini-2.5-flash-image` (Nano Banana). Override with
--model or $NANOBANANA_MODEL (e.g. a Nano Banana Pro id if you have access).
"""
from __future__ import annotations

import argparse
import io
import os
import sys

DEFAULT_MODEL = os.environ.get("NANOBANANA_MODEL", "gemini-2.5-flash-image")
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _client(api_key: str | None):
    try:
        from google import genai  # noqa: WPS433
    except ImportError:
        raise SystemExit("[nano] missing SDK — run: pip install google-genai pillow")
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("[nano] set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)")
    return genai.Client(api_key=key)


def _load_refs(paths: list[str]):
    from PIL import Image  # noqa: WPS433
    imgs = []
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit(f"[nano] reference image not found: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def generate(client, model: str, prompt: str, refs: list, out_path: str) -> None:
    """One call: prompt (+ optional reference images) -> save the first image."""
    from PIL import Image  # noqa: WPS433

    try:
        resp = client.models.generate_content(model=model, contents=[prompt, *refs])
    except Exception as exc:  # noqa: BLE001 — surface the common billing case cleanly
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            raise SystemExit(
                "[nano] 429 quota exceeded. Image generation (Nano Banana) is NOT on the free "
                "tier (limit: 0) — you must enable BILLING on the API key's Google Cloud project, "
                "then re-run.\n"
                "  AI Studio: https://aistudio.google.com/  (open the key's project -> set up billing)\n"
                "  or Cloud Console -> Billing -> link a billing account.\n"
                "  Cost ~$0.039/image (18 images ~ $0.70)."
            )
        raise
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        raise SystemExit(f"[nano] no candidates returned (blocked?): {getattr(resp, 'prompt_feedback', '')}")
    notes = []
    for part in cands[0].content.parts:
        data = getattr(part, "inline_data", None)
        if data and getattr(data, "data", None):
            img = Image.open(io.BytesIO(data.data)).convert("RGB")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            img.save(out_path)
            return
        if getattr(part, "text", None):
            notes.append(part.text)
    # no image part — usually a safety refusal or a text-only reply
    raise SystemExit(f"[nano] no image in response. Model said: {' '.join(notes)[:400] or '(nothing)'}")


def _consistency_suffix() -> str:
    return ("Keep exactly the same character — same face, hairstyle, and outfit — "
            "as in the reference image. Vertical 9:16 composition.")


def run_project(client, model: str, project_dir: str, ref_override: str | None,
                variations: int, overwrite: bool) -> None:
    from shotforge.manifest import load_project

    project = load_project(project_dir)
    ref = ref_override or project.character_ref
    if not ref or not os.path.isfile(ref):
        raise SystemExit(f"[nano] need a reference image (project character_ref={project.character_ref!r}); "
                         f"pass --ref or set cast/character_ref.")
    refs = _load_refs([ref])
    appearance = (project.character or "").strip()
    print(f"[nano] project={project_dir} | model={model} | ref={ref}")

    for shot in project.shots:
        if not shot.frame_prompt.strip():
            print(f"[skip] {shot.id}: no frame_prompt")
            continue
        prompt = ", ".join(x for x in (appearance, shot.frame_prompt.strip()) if x)
        prompt = f"{prompt}. {_consistency_suffix()}"
        base, ext = os.path.splitext(shot.frame)
        for v in range(max(1, variations)):
            out = shot.frame if variations <= 1 else f"{base}_{v + 1}{ext}"
            if os.path.isfile(out) and not overwrite:
                print(f"[skip] {out} exists (use --overwrite)")
                continue
            print(f"[nano] {shot.id}{'' if variations <= 1 else f' v{v + 1}'} -> {out}")
            generate(client, model, prompt, refs, out)
    print("[ok] frames done")


def run_prompts(client, model: str, prompts_file: str, ref_paths: list[str],
                out_dir: str, count: int, overwrite: bool) -> None:
    with open(prompts_file, "r", encoding="utf-8") as fh:
        prompts = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not prompts:
        raise SystemExit(f"[nano] no prompts in {prompts_file}")
    refs = _load_refs(ref_paths) if ref_paths else []
    suffix = (" " + _consistency_suffix()) if refs else " Vertical 9:16 composition."
    os.makedirs(out_dir, exist_ok=True)
    print(f"[nano] prompts={prompts_file} ({len(prompts)}) x{count} | model={model} | refs={ref_paths or 'none'}")

    n = 0
    for i, p in enumerate(prompts, 1):
        for v in range(max(1, count)):
            n += 1
            out = os.path.join(out_dir, f"{n:03d}.png")
            if os.path.isfile(out) and not overwrite:
                print(f"[skip] {out} exists (use --overwrite)")
                continue
            print(f"[nano] {i:02d}/{len(prompts)} v{v + 1} -> {out}")
            generate(client, model, p + suffix, refs, out)
    print(f"[ok] {n} images -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch Nano Banana (Gemini 2.5 Flash Image) generation.")
    ap.add_argument("--project", help="generate a project's shot frames (mode 1)")
    ap.add_argument("--prompts", help="text file, one prompt per line — dataset mode (mode 2)")
    ap.add_argument("--ref", action="append", default=[], help="reference image (repeatable) for character consistency")
    ap.add_argument("--out", default=None, help="output dir (dataset mode)")
    ap.add_argument("--count", type=int, default=1, help="images per prompt (dataset mode)")
    ap.add_argument("--variations", type=int, default=1, help="frames per shot (project mode; >1 -> sN_1, sN_2…)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if bool(args.project) == bool(args.prompts):
        raise SystemExit("pass exactly one of --project (frames) or --prompts (dataset)")

    client = _client(args.api_key)
    if args.project:
        run_project(client, args.model, args.project, args.ref[0] if args.ref else None,
                    args.variations, args.overwrite)
    else:
        out = args.out or "out_nano"
        run_prompts(client, args.model, args.prompts, args.ref, out, args.count, args.overwrite)


if __name__ == "__main__":
    main()
