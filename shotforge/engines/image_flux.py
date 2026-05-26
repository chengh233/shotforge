"""Image engine: FLUX.1-dev text-to-image (+ character LoRA) via ComfyUI — the path
for frames of a character you trained a LoRA for (e.g. wanwan).

Drives a FLUX dev T2I API workflow ($FLUX_WORKFLOW, default comfyui/flux_t2i_api.json,
exported from the GUI). Injects the composed prompt, the shot's LoRA(s) (so the trained
character LoRA is applied + its trigger word activates the identity) and the seed. Pure
text-to-image — refs are ignored (the LoRA carries identity, not a reference image).
Install models with scripts/flux_t2i_setup.py. Prompts: English works best (FLUX's T5).
Reads $COMFY_URL so the Mac can drive a remote Colab ComfyUI.
"""
from __future__ import annotations

import os

from .base import ImageSpec

WORKFLOW = os.environ.get("FLUX_WORKFLOW", "comfyui/flux_t2i_api.json")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


class FluxImageEngine:
    name = "flux"

    def generate(self, spec: ImageSpec) -> None:
        if not os.path.isfile(WORKFLOW):
            raise SystemExit(
                f"[flux] 未找到 FLUX 文生图工作流：{WORKFLOW}\n"
                "  1) 装模型： python scripts/flux_t2i_setup.py\n"
                "  2) GUI 里搭好 FLUX dev 文生图(+Load LoRA) → Export(API) → 存到该路径（或设 $FLUX_WORKFLOW）"
            )
        from shotforge import frames as cf   # reuse the ComfyUI plumbing (queue/wait/download)

        wf = cf.load_workflow(WORKFLOW)

        # prompt -> the longest TextEncode node (the positive CLIPTextEncode). FLUX dev
        # has no real negative (ConditioningZeroOut), so we only set the positive.
        def _tf(n):
            inp = n.get("inputs", {})
            return "text" if "text" in inp else ("prompt" if "prompt" in inp else None)
        encs = [nid for nid, n in wf.items() if "TextEncode" in str(n.get("class_type", "")) and _tf(n)]
        if encs:
            pos = max(encs, key=lambda nid: len(str(wf[nid]["inputs"].get(_tf(wf[nid]), ""))))
            wf[pos]["inputs"][_tf(wf[pos])] = spec.prompt

        # LoRA(s) -> LoraLoader / LoraLoaderModelOnly nodes, in spec order
        lora_nodes = [nid for nid, n in wf.items() if "LoraLoader" in str(n.get("class_type", ""))]
        for nid, lora in zip(lora_nodes, spec.loras):
            inp = wf[nid]["inputs"]
            inp["lora_name"] = lora.path
            for k in ("strength_model", "strength_clip", "strength"):
                if k in inp:
                    inp[k] = lora.weight

        # seed on the sampler
        for n in wf.values():
            inp = n.get("inputs", {})
            ct = str(n.get("class_type", ""))
            if "KSampler" in ct or "SamplerCustom" in ct or "RandomNoise" in ct:
                if "seed" in inp:
                    inp["seed"] = spec.seed
                elif "noise_seed" in inp:
                    inp["noise_seed"] = spec.seed

        outputs = cf._wait(COMFY_URL, cf._queue(COMFY_URL, wf))
        cf._download(COMFY_URL, cf._find_image(outputs), spec.out)


ENGINE = FluxImageEngine()
