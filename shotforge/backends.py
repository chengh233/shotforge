"""Model backends for shotforge.

A *backend* bundles a diffusers image-to-video pipeline class with the
constraints that model imposes — the frame-count rule, the dimension multiple,
and sensible default resolution/steps. The renderer is otherwise model-agnostic:
pick a backend per project via the top-level ``model:`` field in
``project.yaml`` (defaults to ``wan``).

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
    pipeline_cls: str          # diffusers class, e.g. "LTXImageToVideoPipeline"
    default_model_id: str      # HF repo id; override per-run with $I2V_MODEL_ID
    frame_quantum: int         # num_frames must equal frame_quantum * N + 1
    dim_multiple: int          # width and height must be divisible by this
    default_width: int = 480   # 9:16 vertical, the short-drama default
    default_height: int = 832
    default_steps: int = 40
    # Which prompt language the text encoder was trained for. Informational
    # (surfaced in logs/docs, not enforced). See docs/PIPELINE.md.
    prompt_lang: str = "en"


BACKENDS: dict[str, Backend] = {
    # LTX-Video — verified to run on the L4. Its text encoder is T5
    # (English-centric), so write motion prompts in English for this backend.
    "ltx": Backend(
        name="ltx",
        pipeline_cls="LTXImageToVideoPipeline",
        default_model_id="Lightricks/LTX-Video",
        frame_quantum=8,
        dim_multiple=32,
        prompt_lang="en",
    ),
    # Wan 2.2 — the DEFAULT (best for Chinese dramas). Alibaba's model; its text
    # encoder is umT5 (multilingual), so Chinese motion prompts work well here.
    # We default to the 5B TI2V variant because the A14B (MoE) variant does NOT
    # fit a 24GB L4.
    #
    # ⚠️ VERIFY ON THE GPU BOX: the exact repo id, the diffusers class name, and
    # the frame_quantum / dim_multiple below can drift between diffusers
    # releases. If load_pipe can't find the class it prints the available ones —
    # update this entry to match. See docs/PIPELINE.md "切换 / 新增模型".
    "wan": Backend(
        name="wan",
        pipeline_cls="WanImageToVideoPipeline",
        default_model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        frame_quantum=4,
        dim_multiple=16,
        prompt_lang="zh+en",
    ),
}


def get_backend(name: str | None) -> Backend:
    """Look up a backend by name (case-insensitive); default to ``wan``."""
    key = (name or "wan").strip().lower()
    if key not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown model {name!r}; known backends: {known}")
    return BACKENDS[key]
