"""Image-to-video inference, wrapped around a diffusers pipeline.

The default model is LTX-Video; override it with the ``I2V_MODEL_ID`` env var
(e.g. to swap in Wan 2.2). The pipeline is loaded lazily as a process-wide
singleton so the heavy weights are read from disk only once per run.
"""
from __future__ import annotations

import os

import torch

MODEL_ID = os.environ.get("I2V_MODEL_ID", "Lightricks/LTX-Video")

_PIPE = None


def device_dtype() -> tuple[str, "torch.dtype"]:
    """Pick the best available device and a matching dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_pipe():
    """Build the pipeline on first call and reuse it thereafter."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    from diffusers import LTXImageToVideoPipeline

    device, dtype = device_dtype()
    pipe = LTXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

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

    _PIPE = pipe
    return _PIPE


def generate(
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

    pipe = load_pipe()
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
