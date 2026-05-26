"""Talk engine: Sonic (audio-driven portrait) via ComfyUI — ONE pass from a still image +
voice audio to a talking-head video with coordinated head motion + lip sync (replaces the
Wan-motion + LatentSync two-stage, which felt uncoordinated).

Drives a Sonic API workflow ($SONIC_WORKFLOW, default comfyui/sonic_api.json, exported from
the GUI after scripts/sonic_setup.py): uploads the portrait (LoadImage) and the voice
(LoadAudio), sets the seed, queues, downloads the video. Reads $COMFY_URL for a remote box.
"""
from __future__ import annotations

import os
import uuid

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
WORKFLOW = os.environ.get("SONIC_WORKFLOW", "comfyui/sonic_api.json")


class SonicTalkEngine:
    name = "sonic"

    def generate(self, frame: str, audio: str, out: str,
                 seconds: float = 5.0, fps: int = 25, seed: int = 0) -> None:
        if not os.path.isfile(WORKFLOW):
            raise SystemExit(
                f"[sonic] 未找到 Sonic 工作流：{WORKFLOW}\n"
                "  1) 装节点+模型： python scripts/sonic_setup.py\n"
                "  2) GUI 搭好 Sonic（LoadImage + LoadAudio → Sonic → 视频输出）→ Export(API) → 存到该路径"
            )
        from shotforge import comfy

        wf = comfy.load_workflow(WORKFLOW)
        img = [nid for nid, n in wf.items() if n.get("class_type") == "LoadImage"]
        aud = [nid for nid, n in wf.items()
               if "LoadAudio" in str(n.get("class_type", ""))
               or isinstance(n.get("inputs", {}).get("audio"), str)]
        if not img:
            raise SystemExit("[sonic] 工作流里没有 LoadImage 节点")
        if not aud:
            raise SystemExit("[sonic] 工作流里没有音频加载节点（LoadAudio）")
        wf[img[0]]["inputs"]["image"] = comfy._upload_image(COMFY_URL, frame)
        wf[aud[0]]["inputs"]["audio"] = comfy._upload_image(COMFY_URL, audio)  # /upload/image saves any file to input/

        for n in wf.values():
            inp = n.get("inputs", {})
            if "seed" in inp:
                inp["seed"] = int(seed)
            elif "noise_seed" in inp:
                inp["noise_seed"] = int(seed)
            if isinstance(inp.get("duration"), (int, float)):
                inp["duration"] = float(seconds)
            if isinstance(inp.get("fps"), (int, float)):
                inp["fps"] = int(fps)

        outputs = comfy._wait(COMFY_URL, comfy._queue(COMFY_URL, wf, uuid.uuid4().hex))
        comfy._download(COMFY_URL, comfy._find_video(outputs), out)


ENGINE = SonicTalkEngine()
