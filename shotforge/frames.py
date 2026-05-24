"""Generate consistent starting frames with Flux Kontext via ComfyUI — the
`frames` stage.

One reference image of the character (project `character_ref`) keeps every shot
the same person; each shot's **English** `frame_prompt` sets the scene/framing
(Flux Kontext only understands English). One command makes all the frames:

    python -m shotforge.frames --project projects/lasttram
    python run.py frames projects/lasttram

Needs ComfyUI running with the Flux Kontext models (scripts/flux_setup.py) and an
exported Kontext API workflow (default comfyui/flux_kontext_api.json). Injection
points are auto-detected by node class_type — LoadImage (reference), CLIPTextEncode
(prompt), SaveImage (output) — so the exact node ids don't matter.

    python -m shotforge.frames --dump comfyui/flux_kontext_api.json   # list nodes
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


def load_workflow(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ids(wf: dict, class_type: str) -> list[str]:
    return [nid for nid, n in wf.items() if n.get("class_type") == class_type]


def _upload_image(server: str, path: str) -> str:
    with open(path, "rb") as fh:
        resp = requests.post(
            f"{server}/upload/image",
            files={"image": (os.path.basename(path), fh, "application/octet-stream")},
            data={"overwrite": "true"}, timeout=120,
        )
    resp.raise_for_status()
    info = resp.json()
    sub = info.get("subfolder") or ""
    return f"{sub}/{info['name']}" if sub else info["name"]


def _queue(server: str, wf: dict) -> str:
    resp = requests.post(f"{server}/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex}, timeout=120)
    if resp.status_code != 200:
        raise SystemExit(f"[frames] /prompt rejected ({resp.status_code}):\n{resp.text}")
    data = resp.json()
    if data.get("node_errors"):
        raise SystemExit(f"[frames] node errors:\n{json.dumps(data['node_errors'], ensure_ascii=False, indent=2)}")
    return data["prompt_id"]


def _wait(server: str, prompt_id: str, timeout: float = 600.0) -> dict:
    start = time.time()
    while True:
        entry = requests.get(f"{server}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if entry is not None:
            if entry.get("status", {}).get("status_str") == "error":
                raise SystemExit(f"[frames] generation failed: {json.dumps(entry['status'], ensure_ascii=False)[:800]}")
            if entry.get("outputs"):
                return entry["outputs"]
        if time.time() - start > timeout:
            raise SystemExit(f"[frames] timed out after {timeout:.0f}s")
        time.sleep(2)


def _find_image(outputs: dict) -> dict:
    for node_out in outputs.values():
        for items in node_out.values():
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and str(it.get("filename", "")).lower().endswith(_IMG_EXTS):
                        return it
    raise SystemExit(f"[frames] no image in outputs: {json.dumps(outputs, ensure_ascii=False)[:800]}")


def _download(server: str, info: dict, out_path: str) -> None:
    resp = requests.get(f"{server}/view", params={
        "filename": info["filename"], "subfolder": info.get("subfolder", ""), "type": info.get("type", "output"),
    }, timeout=300)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(resp.content)


def render_frame(workflow: dict, server: str, ref_image: str, prompt: str, out_path: str, seed: int = 0) -> None:
    wf = copy.deepcopy(workflow)

    loaders = _ids(wf, "LoadImage")
    if not loaders:
        raise SystemExit("[frames] no LoadImage node in the workflow (need a reference-image input)")
    ref_name = _upload_image(server, ref_image)
    for nid in loaders:
        wf[nid]["inputs"]["image"] = ref_name

    encoders = _ids(wf, "CLIPTextEncode")
    if not encoders:
        raise SystemExit("[frames] no CLIPTextEncode node in the workflow")
    # if several, pick the one with the most existing text (the positive prompt)
    target = max(encoders, key=lambda nid: len(str(wf[nid]["inputs"].get("text", ""))))
    wf[target]["inputs"]["text"] = prompt

    # distinct seed per shot so frames vary even when prompts are close
    for n in wf.values():
        if str(n.get("class_type", "")).startswith("KSampler"):
            inp = n["inputs"]
            if "seed" in inp:
                inp["seed"] = seed
            elif "noise_seed" in inp and inp.get("add_noise", "enable") != "disable":
                inp["noise_seed"] = seed

    outputs = _wait(server, _queue(server, wf))
    _download(server, _find_image(outputs), out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate consistent frames via Flux Kontext (ComfyUI).")
    ap.add_argument("--project")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", default="comfyui/flux_kontext_api.json")
    ap.add_argument("--shot", default=None, help="generate only this shot id")
    ap.add_argument("--dump", default=None, help="just print node ids/classes of a workflow file")
    args = ap.parse_args()

    if args.dump:
        for nid, n in load_workflow(args.dump).items():
            print(f"{nid:>8} {n.get('class_type'):<22} {n.get('_meta', {}).get('title', '')}")
        return

    from .manifest import load_project

    if not args.project:
        raise SystemExit("--project is required")
    project = load_project(args.project)
    if not project.character_ref or not os.path.isfile(project.character_ref):
        raise SystemExit(
            f"[frames] need a character reference image. Set `character_ref:` in project.yaml and put the "
            f"image there (got {project.character_ref!r}). One image of the character keeps all shots consistent."
        )
    if not os.path.isfile(args.workflow):
        raise SystemExit(
            f"[frames] workflow not found: {args.workflow}\n"
            f"        Export ComfyUI's Flux Kontext template as API format and save it there "
            f"(see docs/COMFYUI.md), or pass --workflow <path>."
        )
    workflow = load_workflow(args.workflow)
    print(f"[frames] server={args.comfy_url} | workflow={args.workflow} | ref={project.character_ref}")

    for i, shot in enumerate(project.shots):
        if args.shot and shot.id != args.shot:
            continue
        if not shot.frame_prompt.strip():
            print(f"[skip] {shot.id}: no frame_prompt")
            continue
        # Kontext: the reference image already carries the character — DON'T repeat
        # the appearance in the prompt (that makes it just preserve the reference).
        # The frame_prompt describes only what changes (scene / framing / expression).
        out_path = shot.frame  # already joined to the project dir by load_project
        print(f"[frame] {shot.id} (seed={shot.seed + i}) -> {out_path}")
        render_frame(workflow, args.comfy_url, project.character_ref, shot.frame_prompt, out_path, seed=shot.seed + i)
    print("[ok] frames done")


if __name__ == "__main__":
    main()
