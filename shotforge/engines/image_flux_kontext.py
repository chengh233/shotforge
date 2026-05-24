"""Image engine: FLUX.1 Kontext [dev] via ComfyUI — open-source, runs on the GPU,
multi-reference editing (character + scene + base). Scaffold: wire it to a
multi-image Kontext API workflow. Until then it errors with guidance so you can
fall back to engines.image: nanobanana.

NOTE: FLUX's text encoder is T5 (English-leaning) — prompts compose best in English
on this engine (see compose.yaml / frame `content`).
"""
from __future__ import annotations

import os

from .base import ImageSpec

WORKFLOW = os.environ.get("FLUX_KONTEXT_WORKFLOW", "comfyui/flux_kontext_multi_api.json")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


class FluxKontextEngine:
    name = "flux-kontext"

    def generate(self, spec: ImageSpec) -> None:
        if not os.path.isfile(WORKFLOW):
            raise SystemExit(
                "[flux-kontext] 未配置多图 Kontext 工作流（$FLUX_KONTEXT_WORKFLOW="
                f"{WORKFLOW}）。\n  先在 ComfyUI 里搭一个支持 2~3 张参考图(LoadImage)的 FLUX "
                "Kontext 工作流并导出 API 格式，或先用 engines.image: nanobanana。"
            )
        # Reuse the Kontext driver's plumbing (upload refs -> set prompt -> queue -> download).
        # The multi-image workflow must expose >=len(spec.refs) LoadImage nodes.
        from shotforge import frames as kontext
        wf = kontext.load_workflow(WORKFLOW)
        loaders = kontext._ids(wf, "LoadImage")
        if len(loaders) < len(spec.refs):
            raise SystemExit(f"[flux-kontext] 工作流只有 {len(loaders)} 个 LoadImage，"
                             f"但这一镜需要 {len(spec.refs)} 张参考图。")
        for nid, ref in zip(loaders, spec.refs):
            wf[nid]["inputs"]["image"] = kontext._upload_image(COMFY_URL, ref)
        for nid in kontext._ids(wf, "CLIPTextEncode"):
            wf[nid]["inputs"]["text"] = spec.prompt
            break
        for n in wf.values():
            ct = str(n.get("class_type", ""))
            if ct.startswith("KSampler"):
                inp = n["inputs"]
                if "seed" in inp:
                    inp["seed"] = spec.seed
                elif "noise_seed" in inp:
                    inp["noise_seed"] = spec.seed
        outputs = kontext._wait(COMFY_URL, kontext._queue(COMFY_URL, wf))
        kontext._download(COMFY_URL, kontext._find_image(outputs), spec.out)


ENGINE = FluxKontextEngine()
