"""Generate a character's reference image from its `appearance`, open-source —
an SDXL anime text-to-image workflow in ComfyUI. The result is saved as the
character's `ref` (e.g. characters/<id>/ref.png), which the `frames` stage then
feeds to Flux Kontext to keep every shot consistent.

    python -m shotforge.genref --character yuki
    python run.py genref yuki

Needs ComfyUI running with an anime SDXL checkpoint (Animagine/Illustrious, see
scripts/flux_setup.py) and an exported SDXL txt2img API workflow (default
comfyui/sdxl_txt2img_api.json). Injection points are auto-detected by class_type
(CLIPTextEncode prompt, KSampler seed, SaveImage), so node ids don't matter.

    python -m shotforge.genref --dump comfyui/sdxl_txt2img_api.json   # list nodes
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import uuid

import requests

from .characters import load_character

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ids(wf: dict, class_type: str) -> list[str]:
    return [nid for nid, n in wf.items() if n.get("class_type") == class_type]


def _queue(server: str, wf: dict) -> str:
    resp = requests.post(f"{server}/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex}, timeout=120)
    if resp.status_code != 200:
        raise SystemExit(f"[genref] /prompt rejected ({resp.status_code}):\n{resp.text}")
    data = resp.json()
    if data.get("node_errors"):
        raise SystemExit(f"[genref] node errors:\n{json.dumps(data['node_errors'], ensure_ascii=False, indent=2)}")
    return data["prompt_id"]


def _wait(server: str, prompt_id: str, timeout: float = 300.0) -> dict:
    start = time.time()
    while True:
        entry = requests.get(f"{server}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if entry is not None:
            if entry.get("status", {}).get("status_str") == "error":
                raise SystemExit(f"[genref] failed: {json.dumps(entry['status'], ensure_ascii=False)[:800]}")
            if entry.get("outputs"):
                return entry["outputs"]
        if time.time() - start > timeout:
            raise SystemExit(f"[genref] timed out after {timeout:.0f}s")
        time.sleep(2)


def _download(server: str, outputs: dict, out_path: str) -> None:
    info = None
    for node_out in outputs.values():
        for items in node_out.values():
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and str(it.get("filename", "")).lower().endswith(_IMG_EXTS):
                        info = it
    if info is None:
        raise SystemExit(f"[genref] no image in outputs: {json.dumps(outputs, ensure_ascii=False)[:800]}")
    resp = requests.get(f"{server}/view", params={
        "filename": info["filename"], "subfolder": info.get("subfolder", ""), "type": info.get("type", "output"),
    }, timeout=300)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(resp.content)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a character reference image (SDXL anime, ComfyUI).")
    ap.add_argument("--character", help="character id (characters/<id>) or dir")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", default="comfyui/sdxl_txt2img_api.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default=None, help="just print node ids/classes of a workflow file")
    args = ap.parse_args()

    if args.dump:
        for nid, n in _load(args.dump).items():
            print(f"{nid:>8} {n.get('class_type'):<22} {n.get('_meta', {}).get('title', '')}")
        return

    if not args.character:
        raise SystemExit("--character is required")
    if not os.path.isfile(args.workflow):
        raise SystemExit(f"[genref] workflow not found: {args.workflow}\n"
                         f"        Export ComfyUI's SDXL txt2img template as API format there (see docs/COMFYUI.md).")

    ch = load_character(args.character)
    out_path = ch.ref or os.path.join("characters", ch.id, "ref.png")
    prompt = f"{ch.appearance}, upper body portrait, clean background, 9:16 vertical, masterpiece, best quality"

    wf = copy.deepcopy(_load(args.workflow))
    encs = _ids(wf, "CLIPTextEncode")
    if not encs:
        raise SystemExit("[genref] no CLIPTextEncode node in the workflow")
    target = max(encs, key=lambda nid: len(str(wf[nid]["inputs"].get("text", ""))))
    wf[target]["inputs"]["text"] = prompt
    for nid in _ids(wf, "KSampler"):
        wf[nid]["inputs"]["seed"] = args.seed

    print(f"[genref] {ch.id}: {prompt[:70]}... -> {out_path}")
    _download(args.comfy_url, _wait(args.comfy_url, _queue(args.comfy_url, wf)), out_path)
    print(f"[ok] {ch.id} reference -> {out_path}")


if __name__ == "__main__":
    main()
