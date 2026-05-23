"""Model backends for shotforge.

A *backend* bundles a diffusers image-to-video pipeline class with the
constraints that model imposes — the frame-count rule, the dimension multiple,
the scheduler shift, and sensible default resolution/fps/steps. The renderer is
otherwise model-agnostic: pick a backend per project via the top-level
``model:`` field in ``project.yaml`` (defaults to ``wan``).

Adding a model = adding one entry to ``BACKENDS`` below. The pipeline class is
imported lazily in :func:`shotforge.i2v.load_pipe` — diffusers class names drift
between versions, so we resolve by name at run time and fail loudly (printing
the available class names) if it has moved.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """Everything the renderer needs to know about one I2V model."""

    name: str
    pipeline_cls: str          # diffusers class, e.g. "WanImageToVideoPipeline"
    default_model_id: str      # HF repo id; override per-run with $I2V_MODEL_ID
    frame_quantum: int         # num_frames must equal frame_quantum * N + 1
    dim_multiple: int          # width and height must be divisible by this
    default_width: int = 480   # 9:16 vertical, the short-drama default
    default_height: int = 832
    default_steps: int = 40
    default_fps: int = 24      # the fps the model was trained at (Wan 2.1 = 16)
    flow_shift: float | None = None   # UniPC scheduler shift (Wan: 3.0@480P/5.0@720P)
    # Derive the actual W/H from the starting image's aspect ratio, using
    # default_width*default_height as the area budget (the official Wan I2V
    # approach). Off-aspect/off-bucket dims cause distortion, so prefer this.
    auto_resolution: bool = False
    # Which prompt language the text encoder was trained for. Informational
    # (surfaced in logs/docs, not enforced). See docs/PIPELINE.md.
    prompt_lang: str = "en"


BACKENDS: dict[str, Backend] = {
    # Wan 2.1 I2V 14B (720P) — the DEFAULT, best for Chinese dramas. Proper
    # image-to-video model: WanImageToVideoPipeline auto-loads its CLIP
    # image_encoder. umT5 text encoder → Chinese motion prompts work. 14B in
    # bf16 is ~28GB, so it wants a 40GB+ GPU resident (or offload on smaller —
    # load_pipe decides automatically). Wan 2.1 is trained at 16fps.
    "wan": Backend(
        name="wan",
        pipeline_cls="WanImageToVideoPipeline",
        default_model_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        frame_quantum=4,
        dim_multiple=16,
        default_width=720,
        default_height=1280,
        default_steps=40,
        default_fps=16,
        flow_shift=5.0,       # 5.0 for 720P
        auto_resolution=True, # derive W/H from the image aspect (Wan I2V)
        prompt_lang="zh+en",
    ),
    # Same model family at 480P — lighter/faster, for quick draft renders.
    # Switch a project with `model: wan-480p`.
    "wan-480p": Backend(
        name="wan-480p",
        pipeline_cls="WanImageToVideoPipeline",
        default_model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        frame_quantum=4,
        dim_multiple=16,
        default_width=480,
        default_height=832,
        default_steps=40,
        default_fps=16,
        flow_shift=3.0,       # 3.0 for 480P
        auto_resolution=True, # derive W/H from the image aspect (Wan I2V)
        prompt_lang="zh+en",
    ),
    # LTX-Video — light, fast, English-centric (T5 text encoder). Good for quick
    # iteration or low-VRAM boxes; write motion prompts in English for it.
    "ltx": Backend(
        name="ltx",
        pipeline_cls="LTXImageToVideoPipeline",
        default_model_id="Lightricks/LTX-Video",
        frame_quantum=8,
        dim_multiple=32,
        default_width=480,
        default_height=832,
        default_steps=40,
        default_fps=24,
        flow_shift=None,
        prompt_lang="en",
    ),
}


def get_backend(name: str | None) -> Backend:
    """Look up a backend by name (case-insensitive); default to ``wan``."""
    key = (name or "wan").strip().lower()
    if key not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown model {name!r}; known backends: {known}")
    return BACKENDS[key]
