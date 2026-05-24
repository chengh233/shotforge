"""Generate a project's starting frames by TEXT-TO-IMAGE + a character LoRA —
the `frames-lora` stage. Unlike the Kontext `frames` stage (which edits one
reference and so preserves its composition), this generates each shot fresh, so
every shot gets its OWN framing from its `frame_prompt`, while the character's
trained LoRA keeps her identity/style consistent.

    python -m shotforge.frames_lora --project projects/lasttram
    python run.py frames-lora projects/lasttram

Needs ComfyUI running with the anime SDXL base (Illustrious-XL, scripts/flux_setup.py),
the character's LoRA in ComfyUI/models/loras/<id>.safetensors (scripts/train_lora.py),
and an SDXL+LoRA API workflow (default comfyui/sdxl_lora_t2i_api.json). Injection
points are auto-detected by class_type — LoraLoader (lora), CLIPTextEncode (prompt),
EmptyLatentImage (size), KSampler (seed), SaveImage (output) — so node ids don't matter.

    python -m shotforge.frames_lora --dump comfyui/sdxl_lora_t2i_api.json   # list nodes
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import uuid

import requests

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ids(wf: dict, class_type: str) -> list[str]:
    return [nid for nid, n in wf.items() if n.get("class_type") == class_type]


def _positive_encoders(wf: dict) -> list[str]:
    """CLIPTextEncode nodes feeding a sampler's `positive` input (else the
    longest-text one as a fallback)."""
    positives = set()
    for n in wf.values():
        if str(n.get("class_type", "")).startswith("KSampler"):
            link = n.get("inputs", {}).get("positive")
            if isinstance(link, list) and link:
                positives.add(str(link[0]))
    targets = [p for p in positives if wf.get(p, {}).get("class_type") == "CLIPTextEncode"]
    if targets:
        return targets
    encoders = _ids(wf, "CLIPTextEncode")
    if not encoders:
        raise SystemExit("[frames-lora] no CLIPTextEncode node in the workflow")
    return [max(encoders, key=lambda nid: len(str(wf[nid]["inputs"].get("text", ""))))]


def _queue(server: str, wf: dict) -> str:
    resp = requests.post(f"{server}/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex}, timeout=120)
    if resp.status_code != 200:
        raise SystemExit(f"[frames-lora] /prompt rejected ({resp.status_code}):\n{resp.text}")
    data = resp.json()
    if data.get("node_errors"):
        raise SystemExit(f"[frames-lora] node errors:\n{json.dumps(data['node_errors'], ensure_ascii=False, indent=2)}")
    return data["prompt_id"]


def _wait(server: str, prompt_id: str, timeout: float = 300.0) -> dict:
    start = time.time()
    while True:
        entry = requests.get(f"{server}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if entry is not None:
            if entry.get("status", {}).get("status_str") == "error":
                raise SystemExit(f"[frames-lora] failed: {json.dumps(entry['status'], ensure_ascii=False)[:800]}")
            if entry.get("outputs"):
                return entry["outputs"]
        if time.time() - start > timeout:
            raise SystemExit(f"[frames-lora] timed out after {timeout:.0f}s")
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
        raise SystemExit(f"[frames-lora] no image in outputs: {json.dumps(outputs, ensure_ascii=False)[:800]}")
    resp = requests.get(f"{server}/view", params={
        "filename": info["filename"], "subfolder": info.get("subfolder", ""), "type": info.get("type", "output"),
    }, timeout=300)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(resp.content)


def render(workflow: dict, server: str, prompt: str, out_path: str, *,
           lora_name: str, strength: float, seed: int, width: int, height: int) -> None:
    wf = copy.deepcopy(workflow)

    loras = _ids(wf, "LoraLoader")
    if not loras:
        raise SystemExit("[frames-lora] no LoraLoader node in the workflow (need the character LoRA)")
    for nid in loras:
        wf[nid]["inputs"]["lora_name"] = lora_name
        wf[nid]["inputs"]["strength_model"] = strength
        wf[nid]["inputs"]["strength_clip"] = strength

    for nid in _positive_encoders(wf):
        wf[nid]["inputs"]["text"] = prompt

    for n in wf.values():
        ct = str(n.get("class_type", ""))
        if ct.startswith("KSampler"):
            inp = n["inputs"]
            if "seed" in inp:
                inp["seed"] = seed
            elif "noise_seed" in inp:
                inp["noise_seed"] = seed
        elif ct == "EmptyLatentImage":
            n["inputs"]["width"], n["inputs"]["height"] = width, height

    _download(server, _wait(server, _queue(server, wf)), out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate project frames via SDXL text-to-image + character LoRA.")
    ap.add_argument("--project")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", default="comfyui/sdxl_lora_t2i_api.json")
    ap.add_argument("--shot", default=None, help="generate only this shot id")
    ap.add_argument("--lora", default=None, help="LoRA filename in ComfyUI/models/loras (default <cast id>.safetensors)")
    ap.add_argument("--trigger", default=None, help="LoRA trigger word (default the character's lora_trigger)")
    ap.add_argument("--strength", type=float, default=0.85, help="LoRA strength")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--style", default="masterpiece, best quality, amazing quality, soft cel shading",
                    help="quality/style tags appended to every prompt")
    ap.add_argument("--dump", default=None, help="just print node ids/classes of a workflow file")
    args = ap.parse_args()

    if args.dump:
        for nid, n in _load(args.dump).items():
            print(f"{nid:>8} {n.get('class_type'):<22} {n.get('_meta', {}).get('title', '')}")
        return

    from .characters import load_character
    from .manifest import load_project

    if not args.project:
        raise SystemExit("--project is required")
    if not os.path.isfile(args.workflow):
        raise SystemExit(f"[frames-lora] workflow not found: {args.workflow} (see docs/COMFYUI.md), or pass --workflow")
    project = load_project(args.project)

    # resolve the LoRA + trigger from the cast character (overridable on the CLI)
    trigger, lora_name = args.trigger, args.lora
    if project.cast and (not trigger or not lora_name):
        ch = load_character(project.cast)
        trigger = trigger or ch.lora_trigger
        lora_name = lora_name or f"{ch.id}.safetensors"
    if not lora_name:
        raise SystemExit("[frames-lora] no LoRA: set `cast:` in project.yaml or pass --lora <file> (and --trigger)")
    trigger = (trigger or "").strip()

    workflow = _load(args.workflow)
    print(f"[frames-lora] server={args.comfy_url} | lora={lora_name} (x{args.strength}) | trigger={trigger!r}")

    for i, shot in enumerate(project.shots):
        if args.shot and shot.id != args.shot:
            continue
        if not shot.frame_prompt.strip():
            print(f"[skip] {shot.id}: no frame_prompt")
            continue
        parts = [p for p in (trigger, shot.frame_prompt.strip(), args.style.strip()) if p]
        prompt = ", ".join(parts)
        seed = shot.seed + i
        print(f"[frame] {shot.id} (seed={seed}) -> {shot.frame}")
        render(workflow, args.comfy_url, prompt, shot.frame,
               lora_name=lora_name, strength=args.strength, seed=seed, width=args.width, height=args.height)
    print("[ok] frames-lora done")


if __name__ == "__main__":
    main()
