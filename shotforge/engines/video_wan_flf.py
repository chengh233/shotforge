"""Video engine: Wan FIRST-LAST-FRAME (FLF2V) via ComfyUI. Takes a START and an
END keyframe and fills the motion BETWEEN them — far more directed and stable than
single-frame I2V (the model can't wander; it must land on the end frame). Best
stability when the two keyframes are consistent (generate the end as an edit of the
start) and close together (gentle motion).

Needs a Wan FLF2V workflow exported from ComfyUI ($WAN_FLF_WORKFLOW, default
comfyui/wan_flf_api.json) with TWO LoadImage inputs. Start/end are mapped by
tracing the WanFirstLastFrameToVideo node's start_image / end_image links.
"""
from __future__ import annotations

import os

from .base import MotionSpec

WORKFLOW = os.environ.get("WAN_FLF_WORKFLOW", "comfyui/wan_flf_api.json")
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


class WanFLFEngine:
    name = "wan-flf"

    def animate(self, spec: MotionSpec) -> None:
        if not os.path.isfile(WORKFLOW):
            raise SystemExit(
                f"[wan-flf] 未找到首尾帧工作流：{WORKFLOW}\n"
                "  在 ComfyUI 装好 Wan 首尾帧(FLF2V)模型、导出含【两个 LoadImage(首/尾)】的 API 工作流到该路径。"
            )
        from shotforge import comfy
        from shotforge.manifest import frames_for

        wf = comfy.load_workflow(WORKFLOW)
        loaders = [nid for nid, n in wf.items() if n.get("class_type") == "LoadImage"]
        if len(loaders) < 2:
            raise SystemExit(f"[wan-flf] 工作流只有 {len(loaders)} 个 LoadImage，需要 2 个（首帧+尾帧）。")
        # robust start/end mapping: trace the FLF node's start_image / end_image links
        start_nid = end_nid = None
        flf = next((n for n in wf.values() if n.get("class_type") == "WanFirstLastFrameToVideo"), None)
        if flf:
            si, ei = flf["inputs"].get("start_image"), flf["inputs"].get("end_image")
            start_nid = str(si[0]) if isinstance(si, list) else None
            end_nid = str(ei[0]) if isinstance(ei, list) else None
        if not (start_nid and end_nid):   # fallback: title keyword, else order
            def _t(nid):
                return str(wf[nid].get("_meta", {}).get("title", "")).lower()
            end_nid = end_nid or next((n for n in loaders if any(k in _t(n) for k in ("end", "last", "尾", "结束"))), loaders[1])
            start_nid = start_nid or next((n for n in loaders if n != end_nid), loaders[0])
        wf[start_nid]["inputs"]["image"] = comfy._upload_image(COMFY_URL, spec.frame)
        wf[end_nid]["inputs"]["image"] = comfy._upload_image(COMFY_URL, spec.end_frame or spec.frame)

        encs = [nid for nid, n in wf.items() if n.get("class_type") == "CLIPTextEncode"]
        if encs:
            pos = max(encs, key=lambda nid: len(str(wf[nid]["inputs"].get("text", ""))))
            wf[pos]["inputs"]["text"] = spec.prompt
            for nid in encs:
                if nid != pos:
                    wf[nid]["inputs"]["text"] = spec.negative

        w, h = (spec.width, spec.height) if (spec.width and spec.height) else (480, 832)
        length = frames_for(spec.seconds, spec.fps, 4)
        for n in wf.values():
            inp = n.get("inputs", {})
            if "width" in inp and "height" in inp:
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

        import uuid
        outputs = comfy._wait(COMFY_URL, comfy._queue(COMFY_URL, wf, uuid.uuid4().hex))
        comfy._download(COMFY_URL, comfy._find_video(outputs), spec.out)


ENGINE = WanFLFEngine()
