"""Render shots through a running ComfyUI server's HTTP API (engine: comfy).

diffusers' Wan I2V melts/garbles; ComfyUI's native Wan implementation is the
quality path (see docs/COMFYUI.md). This drives an API-format workflow exported
from ComfyUI: upload the start frame, inject per-shot params, queue, wait, then
download the resulting video.

ComfyUI must be running and reachable (same box as shotforge by default):
    python ComfyUI/main.py --listen 0.0.0.0 --port 8188

The NODE_* ids below come from comfyui/wan_i2v_api.json. If you re-export a
changed workflow, re-check them — `python -m shotforge.comfy <workflow.json>`
dumps every node id + class so you can find the new ones.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid

import requests

# ComfyUI's log (KSampler tqdm progress goes here). Override with $COMFY_LOG.
LOG_PATH = os.environ.get("COMFY_LOG", "/content/comfyui.log")
_PCT_RE = re.compile(r"(\d{1,3})%\|")     # tqdm: "  45%|████ | 9/20 ..."
_FRAC_RE = re.compile(r"(\d+)/(\d+)")

# Injection points in comfyui/wan_i2v_api.json:
NODE_IMAGE = "97"          # LoadImage.image (the uploaded start frame)
NODE_POSITIVE = "129:93"   # CLIPTextEncode (positive) .text
NODE_NEGATIVE = "129:89"   # CLIPTextEncode (negative) .text
NODE_WANI2V = "129:98"     # WanImageToVideo .width / .height
NODE_DURATION = "129:161"  # PrimitiveFloat duration in seconds .value (frames = floor(dur*fps+1))
NODE_FPS = "129:162"       # PrimitiveFloat fps .value
NODE_SEED = "129:86"       # KSamplerAdvanced (the noise-adding sampler) .noise_seed

_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".gif", ".webp")


def load_workflow(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def derive_dims(frame_path: str, area: int, multiple: int = 16) -> tuple[int, int]:
    """Aspect-correct width/height within an area budget, snapped to ``multiple``."""
    from PIL import Image

    with Image.open(frame_path) as im:
        iw, ih = im.size
    ar = ih / iw
    h = max(multiple, int(round((area * ar) ** 0.5)) // multiple * multiple)
    w = max(multiple, int(round((area / ar) ** 0.5)) // multiple * multiple)
    return w, h


def _upload_image(server: str, path: str) -> str:
    """Upload a start frame to ComfyUI's input dir; return the name to reference."""
    with open(path, "rb") as fh:
        resp = requests.post(
            f"{server}/upload/image",
            files={"image": (os.path.basename(path), fh, "application/octet-stream")},
            data={"overwrite": "true"},
            timeout=120,
        )
    resp.raise_for_status()
    info = resp.json()
    sub = info.get("subfolder") or ""
    return f"{sub}/{info['name']}" if sub else info["name"]


def _queue(server: str, workflow: dict, client_id: str) -> str:
    resp = requests.post(
        f"{server}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=120
    )
    if resp.status_code != 200:
        # surface ComfyUI's validation errors (missing models, bad inputs, ...)
        raise SystemExit(f"[comfy] /prompt rejected ({resp.status_code}):\n{resp.text}")
    data = resp.json()
    if data.get("node_errors"):
        raise SystemExit(f"[comfy] node errors:\n{json.dumps(data['node_errors'], ensure_ascii=False, indent=2)}")
    return data["prompt_id"]


def _log_progress() -> str | None:
    """Latest KSampler step progress from ComfyUI's log tail, e.g. '45% (9/20)'.
    Wan 2.2 runs two sampler passes (high- then low-noise expert), so it climbs
    0→100% twice per shot, then the VAE decode runs."""
    try:
        with open(LOG_PATH, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4096))
            chunk = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    for frag in reversed(re.split(r"[\r\n]", chunk)):  # tqdm updates in place with \r
        m = _PCT_RE.search(frag)
        if m:
            f = _FRAC_RE.search(frag)
            return f"{m.group(1)}%" + (f" ({f.group(0)})" if f else "")
    return None


def _wait(server: str, prompt_id: str, timeout: float = 3600.0) -> dict:
    start = time.time()
    beat = 0.0
    while True:
        hist = requests.get(f"{server}/history/{prompt_id}", timeout=30).json()
        entry = hist.get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise SystemExit(f"[comfy] render failed: {json.dumps(status, ensure_ascii=False)[:1000]}")
            if entry.get("outputs"):
                print(f"[comfy] done in {time.time() - start:.0f}s")
                return entry["outputs"]
        elapsed = time.time() - start
        if elapsed - beat >= 10:
            prog = _log_progress()
            tail = f" | step {prog}" if prog else f"; `tail {LOG_PATH}` for step progress"
            print(f"[comfy]   …rendering ({elapsed:.0f}s{tail})")
            beat = elapsed
        if elapsed > timeout:
            raise SystemExit(f"[comfy] timed out after {timeout:.0f}s waiting for {prompt_id}")
        time.sleep(2)


def _find_video(outputs: dict) -> dict:
    for node_out in outputs.values():
        for items in node_out.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and str(it.get("filename", "")).lower().endswith(_VIDEO_EXTS):
                    return it
    raise SystemExit(f"[comfy] no video file in outputs: {json.dumps(outputs, ensure_ascii=False)[:1000]}")


def _download(server: str, fileinfo: dict, out_path: str) -> None:
    params = {
        "filename": fileinfo["filename"],
        "subfolder": fileinfo.get("subfolder", ""),
        "type": fileinfo.get("type", "output"),
    }
    resp = requests.get(f"{server}/view", params=params, timeout=600)
    resp.raise_for_status()
    with open(out_path, "wb") as fh:
        fh.write(resp.content)


def render_shot(
    workflow: dict,
    server: str,
    *,
    frame_path: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seconds: float,
    fps: int,
    seed: int,
    out_path: str,
) -> None:
    """Render one shot via ComfyUI and write the result to ``out_path``."""
    wf = copy.deepcopy(workflow)
    wf[NODE_IMAGE]["inputs"]["image"] = _upload_image(server, frame_path)
    wf[NODE_POSITIVE]["inputs"]["text"] = prompt
    wf[NODE_NEGATIVE]["inputs"]["text"] = negative
    wf[NODE_WANI2V]["inputs"]["width"] = int(width)
    wf[NODE_WANI2V]["inputs"]["height"] = int(height)
    wf[NODE_DURATION]["inputs"]["value"] = float(seconds)
    wf[NODE_FPS]["inputs"]["value"] = float(fps)
    wf[NODE_SEED]["inputs"]["noise_seed"] = int(seed)

    prompt_id = _queue(server, wf, uuid.uuid4().hex)
    print(f"[comfy] queued {prompt_id}; rendering...")
    outputs = _wait(server, prompt_id)
    _download(server, _find_video(outputs), out_path)


if __name__ == "__main__":  # dump node ids/classes to (re)locate injection points
    import sys

    wf = load_workflow(sys.argv[1] if len(sys.argv) > 1 else "comfyui/wan_i2v_api.json")
    for nid, node in wf.items():
        print(f"{nid:>8} {node.get('class_type'):<24} {node.get('_meta', {}).get('title', '')}")
