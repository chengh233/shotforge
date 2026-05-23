"""Image-to-video inference, wrapped around a diffusers pipeline.

The model is chosen per project via the ``model:`` field in project.yaml,
resolved to a backend in :mod:`shotforge.backends`. Override the concrete
checkpoint for a single run with the ``I2V_MODEL_ID`` env var. The pipeline is
loaded lazily and cached as a process-wide singleton (keyed by class +
checkpoint) so the heavy weights are read from disk only once per run.

Memory strategy is chosen by VRAM: big GPUs (>=40GB) keep the model resident
(fastest); smaller ones stream it via cpu offload. Override with $I2V_OFFLOAD.
"""
from __future__ import annotations

import importlib
import os

import torch

from .backends import Backend

# Optional per-run override of the concrete checkpoint, applied to whichever
# backend is selected. Leave unset to use the backend's default_model_id.
MODEL_ID_OVERRIDE = os.environ.get("I2V_MODEL_ID")

# Below this much VRAM we stream weights with cpu offload instead of pinning
# the whole model. 14B in bf16 (~28GB) doesn't fit a 24GB card resident.
_OFFLOAD_VRAM_THRESHOLD = 40 * (1024 ** 3)

_PIPE = None
_PIPE_KEY: tuple[str, str] | None = None


def device_dtype() -> tuple[str, "torch.dtype"]:
    """Pick the best available device and a matching dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _should_offload() -> bool:
    """Decide whether to cpu-offload: forced by $I2V_OFFLOAD, else by VRAM."""
    val = os.environ.get("I2V_OFFLOAD", "auto").strip().lower()
    if val in ("1", "true", "on"):
        return True
    if val in ("0", "false", "off"):
        return False
    try:  # auto: only offload when VRAM is tight
        return torch.cuda.get_device_properties(0).total_memory < _OFFLOAD_VRAM_THRESHOLD
    except Exception:
        return True


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
    # from_pretrained pulls every component listed in the repo (for Wan I2V that
    # includes the CLIP image_encoder), so a plain load is enough.
    pipe = PipelineClass.from_pretrained(model_id, torch_dtype=dtype)

    # Video VAEs are precision-sensitive: in fp16/bf16 the temporal decode can
    # overflow to NaN, leaving every frame after the first one blank. Decode the
    # VAE (and Wan's CLIP image_encoder) in fp32. Set $I2V_VAE_DTYPE=bf16 to
    # trade it back for memory once you've confirmed clean decodes.
    if os.environ.get("I2V_VAE_DTYPE", "fp32").lower() in ("fp32", "float32"):
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.to(torch.float32)
            print("[vae] dtype=float32 (set I2V_VAE_DTYPE=bf16 to use the model dtype)")
        if getattr(pipe, "image_encoder", None) is not None:
            pipe.image_encoder.to(torch.float32)

    # Wan recommends a resolution-dependent flow shift on a UniPC scheduler.
    if backend.flow_shift is not None:
        try:
            from diffusers import UniPCMultistepScheduler

            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=backend.flow_shift
            )
            print(f"[sched] UniPC flow_shift={backend.flow_shift}")
        except Exception as exc:  # non-fatal: keep the default scheduler
            print(f"[warn] could not set flow_shift: {exc}")

    if device == "cuda":
        if _should_offload():
            # Stream weights through VRAM instead of pinning the whole model.
            pipe.enable_model_cpu_offload()
            print("[mem] model cpu offload ON")
            # VAE tiling keeps decode memory flat but has caused blank/garbled
            # frames with some video VAEs, so it's opt-in via $I2V_VAE_TILING=1.
            if os.environ.get("I2V_VAE_TILING") == "1" and getattr(pipe, "vae", None) is not None:
                try:
                    pipe.vae.enable_tiling()
                    print("[vae] tiling enabled")
                except Exception as exc:  # best effort; not all builds support it
                    print(f"[warn] vae tiling unavailable: {exc}")
        else:
            pipe.to(device)
            print("[mem] model resident on cuda (no offload)")
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
