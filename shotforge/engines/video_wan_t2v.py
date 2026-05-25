"""Video engine: Wan 2.2 TEXT-to-video via ComfyUI (no starting frame).

The model generates the whole clip from text and keeps consistency WITHIN a clip.
Cross-clip character consistency is weak (each clip is independent) — the composer
repeats the character description verbatim and the project fixes the seed to
maximize similarity, but it won't match faces like I2V-from-a-fixed-frame.

Needs a Wan T2V API workflow exported from ComfyUI ($WAN_T2V_WORKFLOW, default
comfyui/wan_t2v_api.json) AND the Wan 2.2 T2V models installed (different from the
I2V models). Nodes are auto-detected by class_type, so node ids don't matter.
"""
from __future__ import annotations

import copy
import os
import uuid

from .base import MotionSpec

WORKFLOW = os.environ.get("WAN_T2V_WORKFLOW", "comfyui/wan_t2v_api.json")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


class WanT2VEngine:
    name = "wan-t2v"

    def animate(self, spec: MotionSpec) -> None:
        if not os.path.isfile(WORKFLOW):
            raise SystemExit(
                f"[wan-t2v] 未找到 T2V 工作流：{WORKFLOW}\n"
                "  1) 装 T2V 模型： python scripts/wan_t2v_setup.py\n"
                "  2) 取工作流：ComfyUI GUI 打开「Wan 2.2 14B Text to Video」模板 → Export(API) → 存到该路径\n"
                "  （或设 $WAN_T2V_WORKFLOW 指向你的工作流）"
            )
        from shotforge import comfy
        from shotforge.manifest import frames_for

        wf = copy.deepcopy(comfy.load_workflow(WORKFLOW))
        # positive = longest-text CLIPTextEncode; the rest get the negative
        encs = [nid for nid, n in wf.items() if n.get("class_type") == "CLIPTextEncode"]
        if not encs:
            raise SystemExit("[wan-t2v] 工作流里没有 CLIPTextEncode 节点")
        pos = max(encs, key=lambda nid: len(str(wf[nid]["inputs"].get("text", ""))))
        wf[pos]["inputs"]["text"] = spec.prompt
        for nid in encs:
            if nid != pos:
                wf[nid]["inputs"]["text"] = spec.negative

        w, h = spec.width or 480, spec.height or 832
        length = frames_for(spec.seconds, spec.fps, 4)  # Wan quantum = 4
        for n in wf.values():
            inp = n.get("inputs", {})
            if "width" in inp and "height" in inp:     # the empty latent-video node
                inp["width"], inp["height"] = w, h
            if "length" in inp:
                inp["length"] = length
            elif "num_frames" in inp:
                inp["num_frames"] = length
            if str(n.get("class_type", "")).startswith("KSampler"):
                if "seed" in inp:
                    inp["seed"] = spec.seed
                elif "noise_seed" in inp:
                    inp["noise_seed"] = spec.seed

        outputs = comfy._wait(COMFY_URL, comfy._queue(COMFY_URL, wf, uuid.uuid4().hex))
        comfy._download(COMFY_URL, comfy._find_video(outputs), spec.out)


ENGINE = WanT2VEngine()
