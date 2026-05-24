"""Video engine: Wan 2.2 I2V via a running ComfyUI server. Thin adapter over the
proven shotforge/comfy.py driver. Content-free; the motion prompt comes from the
composer. Derives aspect-correct dims from the frame within the shot's area budget.
"""
from __future__ import annotations

import os

from .base import MotionSpec

WORKFLOW = os.environ.get("WAN_WORKFLOW", "comfyui/wan_i2v_api.json")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
_DIM_MULTIPLE = 16  # Wan


class WanEngine:
    name = "wan"

    def animate(self, spec: MotionSpec) -> None:
        from shotforge import comfy
        wf = comfy.load_workflow(WORKFLOW)
        if spec.width and spec.height:
            w, h = comfy.derive_dims(spec.frame, spec.width * spec.height, _DIM_MULTIPLE)
        else:
            w, h = comfy.derive_dims(spec.frame, 480 * 832, _DIM_MULTIPLE)
        comfy.render_shot(
            wf, COMFY_URL,
            frame_path=spec.frame, prompt=spec.prompt, negative=spec.negative,
            width=w, height=h, seconds=spec.seconds, fps=spec.fps, seed=spec.seed,
            out_path=spec.out,
        )


ENGINE = WanEngine()
