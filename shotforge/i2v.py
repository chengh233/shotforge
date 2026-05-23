"""Image-to-video inference, wrapped around a diffusers pipeline.

The model is chosen per project via the ``model:`` field in project.yaml,
resolved to a backend in :mod:`shotforge.backends`. Override the concrete
checkpoint for a single run with the ``I2V_MODEL_ID`` env var (e.g. to pin a
quantized or fine-tuned checkpoint of the same family). The pipeline is loaded
lazily and cached as a process-wide singleton (keyed by class + checkpoint) so
the heavy weights are read from disk only once per run.
"""
from __future__ import annotations

import importlib
import os

import torch

from .backends import Backend

# Optional per-run override of the concrete checkpoint, applied to whichever
# backend is selected. Leave unset to use the backend's default_model_id.
MODEL_ID_OVERRIDE = os.environ.get("I2V_MODEL_ID")

_PIPE = None
_PIPE_KEY: tuple[str, str] | None = None


def device_dtype() -> tuple[str, "torch.dtype"]:
    """Pick the best available device and a matching dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _resolve_pipeline_class(name: str):
    """Import a diffusers pipeline class by name, failing loudly if it moved.

    diffusers renames pipeline classes between versions; rather than a cryptic
    ImportError we list the I2V classes the installed version actually exposes
    so backends.py can be corrected.
    """
    diffusers = importlib.import_module("diffusers")
    try:
        return getattr(diffusers, name)
    except AttributeError:
        candidates = sorted(n for n in dir(diffusers) if "ImageToVideo" in n)
        raise SystemExit(
            f"[error] diffusers {diffusers.__version__} has no class {name!r}.\n"
            f"        Pipeline class names drift between versions — update the "
            f"backend in shotforge/backends.py.\n"
            f"        Available image-to-video pipelines: {candidates or '(none found)'}"
        )


def load_pipe(backend: Backend):
    """Build the pipeline for ``backend`` on first use; reuse it thereafter."""
    global _PIPE, _PIPE_KEY

    model_id = MODEL_ID_OVERRIDE or backend.default_model_id
    key = (backend.pipeline_cls, model_id)
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE

    PipelineClass = _resolve_pipeline_class(backend.pipeline_cls)
    device, dtype = device_dtype()
    print(f"[model] {backend.name}: {backend.pipeline_cls} <- {model_id}")
    pipe = PipelineClass.from_pretrained(model_id, torch_dtype=dtype)

    if device == "cuda":
        # Stream weights through the 24GB L4 instead of pinning the whole model
        # in VRAM, and tile the VAE to keep decode memory flat.
        pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception as exc:  # best effort; not all builds support tiling
            print(f"[warn] vae tiling unavailable: {exc}")
    else:
        pipe.to(device)

    _PIPE, _PIPE_KEY = pipe, key
    return _PIPE


def generate(
    backend: Backend,
    frame_path: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    seed: int,
):
    """Render one shot from a starting frame and return its list of PIL frames."""
    from diffusers.utils import load_image

    pipe = load_pipe(backend)
    device, _ = device_dtype()

    image = load_image(frame_path)
    # mps has no Generator implementation; seed on cpu there.
    gen_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    result = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        generator=generator,
    )
    return result.frames[0]
