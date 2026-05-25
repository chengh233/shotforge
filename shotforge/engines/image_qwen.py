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

WORKFLOW = os.environ.get("QWEN_WORKFLOW", "comfyui/qwen_image_api.json")            # text-to-image
EDIT_WORKFLOW = os.environ.get("QWEN_EDIT_WORKFLOW", "comfyui/qwen_edit_api.json")   # reference edit
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


class QwenImageEngine:
    name = "qwen"

    def generate(self, spec: ImageSpec) -> None:
        # refs present (e.g. an end-frame edited from the start) -> Qwen-Image-Edit;
        # no refs -> plain text-to-image.
        path = EDIT_WORKFLOW if spec.refs else WORKFLOW
        if not os.path.isfile(path):
            kind = "Qwen-Image-Edit（参考编辑）" if spec.refs else "Qwen-Image（文生图）"
            raise SystemExit(
                f"[qwen] 未找到{kind}工作流：{path}\n"
                "  1) 装模型： python scripts/qwen_setup.py\n"
                f"  2) ComfyUI 打开 {kind} 模板 → Export(API) → 存到该路径（或设环境变量）"
            )
        from shotforge import frames as cf   # reuse the ComfyUI plumbing

        wf = cf.load_workflow(path)
        # refs -> LoadImage nodes (only when it's an Edit workflow with image inputs)
        loaders = cf._ids(wf, "LoadImage")
        if loaders and spec.refs:
            for nid, ref in zip(loaders, spec.refs):
                wf[nid]["inputs"]["image"] = cf._upload_image(COMFY_URL, ref)
        # set the prompt on the text-encode node(s). Handles both CLIPTextEncode (T2I)
        # and TextEncodeQwenImageEdit / *Plus (Edit), whose field may be text or prompt.
        def _tf(n):
            inp = n.get("inputs", {})
            return "text" if "text" in inp else ("prompt" if "prompt" in inp else None)
        encs = [nid for nid, n in wf.items() if "TextEncode" in str(n.get("class_type", "")) and _tf(n)]
        if encs:
            pos = max(encs, key=lambda nid: len(str(wf[nid]["inputs"].get(_tf(wf[nid]), ""))))
            wf[pos]["inputs"][_tf(wf[pos])] = spec.prompt
            for nid in encs:
                if nid != pos:
                    wf[nid]["inputs"][_tf(wf[nid])] = spec.negative
        # output size + seed. Qwen wants >=~1024px; the video engine downscales the
        # frame to its own (e.g. 480p) budget, so generate the still bigger for quality.
        w, h = spec.width, spec.height
        if max(w, h) < 1024:
            f = 1024.0 / max(w, h)
            w, h = int(round(w * f / 16)) * 16, int(round(h * f / 16)) * 16
        for n in wf.values():
            inp = n.get("inputs", {})
            if "width" in inp and "height" in inp:
                inp["width"], inp["height"] = w, h
            if str(n.get("class_type", "")).startswith("KSampler"):
                if "seed" in inp:
                    inp["seed"] = spec.seed
                elif "noise_seed" in inp:
                    inp["noise_seed"] = spec.seed

        outputs = cf._wait(COMFY_URL, cf._queue(COMFY_URL, wf))
        cf._download(COMFY_URL, cf._find_image(outputs), spec.out)


ENGINE = QwenImageEngine()
