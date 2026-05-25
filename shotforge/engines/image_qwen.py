"""Image engine: Qwen-Image (Alibaba, 20B MMDiT) via ComfyUI — open-source, strong
photorealism + Chinese text/faces. Runs on the GPU (A100/H100).

Drives a Qwen-Image API workflow ($QWEN_WORKFLOW, default comfyui/qwen_image_api.json).
If the workflow has LoadImage nodes (i.e. you exported the Qwen-Image-EDIT template)
and the shot carries refs, the refs are uploaded for reference-conditioned generation;
otherwise it's plain text-to-image. Nodes auto-detected by class_type — install
models with scripts/qwen_setup.py, export the template as API. Prompts: 中文 or English.
"""
from __future__ import annotations

import os

from .base import ImageSpec

WORKFLOW = os.environ.get("QWEN_WORKFLOW", "comfyui/qwen_image_api.json")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


class QwenImageEngine:
    name = "qwen"

    def generate(self, spec: ImageSpec) -> None:
        if not os.path.isfile(WORKFLOW):
            raise SystemExit(
                f"[qwen] 未找到工作流：{WORKFLOW}\n"
                "  1) 装模型： python scripts/qwen_setup.py\n"
                "  2) 取工作流：ComfyUI GUI 打开 Qwen-Image（或 Qwen-Image-Edit）模板 → Export(API) → 存到该路径\n"
                "  （或设 $QWEN_WORKFLOW）"
            )
        from shotforge import frames as cf   # reuse the ComfyUI plumbing

        wf = cf.load_workflow(WORKFLOW)
        # refs -> LoadImage nodes (only when it's an Edit workflow with image inputs)
        loaders = cf._ids(wf, "LoadImage")
        if loaders and spec.refs:
            for nid, ref in zip(loaders, spec.refs):
                wf[nid]["inputs"]["image"] = cf._upload_image(COMFY_URL, ref)
        # prompt = longest-text CLIPTextEncode (positive); the rest get the negative
        encs = cf._ids(wf, "CLIPTextEncode")
        if encs:
            pos = max(encs, key=lambda nid: len(str(wf[nid]["inputs"].get("text", ""))))
            wf[pos]["inputs"]["text"] = spec.prompt
            for nid in encs:
                if nid != pos:
                    wf[nid]["inputs"]["text"] = spec.negative
        # output size + seed
        for n in wf.values():
            inp = n.get("inputs", {})
            if "width" in inp and "height" in inp:
                inp["width"], inp["height"] = spec.width, spec.height
            if str(n.get("class_type", "")).startswith("KSampler"):
                if "seed" in inp:
                    inp["seed"] = spec.seed
                elif "noise_seed" in inp:
                    inp["noise_seed"] = spec.seed

        outputs = cf._wait(COMFY_URL, cf._queue(COMFY_URL, wf))
        cf._download(COMFY_URL, cf._find_image(outputs), spec.out)


ENGINE = QwenImageEngine()
