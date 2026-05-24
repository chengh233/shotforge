#!/usr/bin/env python3
"""Train a character LoRA (SDXL / Illustrious-XL) with kohya sd-scripts, on the
GPU box (A100). Reads a folder of images of ONE character and produces a LoRA
that makes the anime SDXL base render that character consistently — which
`frames-lora` then uses to generate each shot fresh with its own framing.

    python scripts/train_lora.py --character yuki
    python run.py train yuki

Inputs : characters/<id>/dataset/*.png|jpg   (e.g. ~15-20 varied images; build them
         with tools/nanobanana.py or `python run.py frames --character <id> --prompts ...`)
Outputs: ComfyUI/models/loras/<id>.safetensors   (so ComfyUI/`frames-lora` can load it)
         characters/<id>/<id>.safetensors          (a copy to keep / back up — gitignored)

kohya is installed into an isolated venv (reusing the system torch via
--system-site-packages) so it can't disturb ComfyUI's package versions. This is
the heaviest step and the one most likely to need a dep tweak on a given Colab
image; the command it runs is printed so you can rerun/adjust by hand.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFY", "/content/ComfyUI")
SD_SCRIPTS = os.environ.get("SD_SCRIPTS", "/content/sd-scripts")
VENV = os.environ.get("LORA_VENV", "/content/lora-venv")
VENV_PY = os.path.join(VENV, "bin", "python")
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def sh(*cmd: str, **kw) -> None:
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), check=True, **kw)


def ensure_sd_scripts() -> None:
    if os.path.isfile(os.path.join(SD_SCRIPTS, "sdxl_train_network.py")):
        print(f"[train] sd-scripts present at {SD_SCRIPTS}")
        return
    sh("git", "clone", "--depth", "1", "https://github.com/kohya-ss/sd-scripts", SD_SCRIPTS)


def ensure_venv() -> None:
    """Isolated venv that reuses the system torch (no multi-GB reinstall) but
    installs kohya's other deps without touching ComfyUI's env."""
    if not os.path.isfile(VENV_PY):
        sh(sys.executable, "-m", "venv", "--system-site-packages", VENV)
        sh(VENV_PY, "-m", "pip", "install", "-q", "-U", "pip")
    # kohya's requirements minus torch/xformers (reused from system) and the editable self-install
    req = os.path.join(SD_SCRIPTS, "requirements.txt")
    pkgs: list[str] = []
    if os.path.isfile(req):
        with open(req, encoding="utf-8") as fh:
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#") or s in {".", "-e ."} or s.startswith("-e"):
                    continue
                low = s.lower()
                if low.startswith(("torch", "torchvision", "xformers")):
                    continue
                pkgs.append(s)
    # belt-and-suspenders: things kohya always needs for SDXL + 8bit optim
    pkgs += ["accelerate", "transformers", "diffusers", "safetensors", "bitsandbytes", "toml", "voluptuous"]
    sh(VENV_PY, "-m", "pip", "install", "-q", *pkgs)


def build_dataset(char_id: str, data_dir: str, trigger: str, repeats: int) -> str:
    """Lay the images out the way kohya expects: <root>/<repeats>_<trigger>/ with a
    .txt caption (the trigger) beside each image. Returns the dataset ROOT."""
    imgs = sorted(p for p in glob.glob(os.path.join(data_dir, "*")) if p.lower().endswith(_IMG_EXTS))
    if not imgs:
        raise SystemExit(f"[train] no images in {data_dir} — generate a dataset first "
                         f"(tools/nanobanana.py or `run.py frames --character {char_id} --prompts ...`)")
    root = os.path.join("/content", f"lora_{char_id}")
    concept = os.path.join(root, f"{repeats}_{trigger}")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(concept, exist_ok=True)
    caption = f"{trigger}, 1girl, solo"
    for i, src in enumerate(imgs, 1):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(concept, f"{i:03d}{ext}")
        shutil.copy(src, dst)
        with open(os.path.splitext(dst)[0] + ".txt", "w", encoding="utf-8") as fh:
            fh.write(caption)
    print(f"[train] {len(imgs)} images x{repeats} repeats -> {concept} (caption: {caption!r})")
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a character LoRA with kohya sd-scripts (SDXL).")
    ap.add_argument("--character", required=True, help="character id (characters/<id>)")
    ap.add_argument("--base", default=os.path.join(COMFY, "models/checkpoints/Illustrious-XL-v1.0.safetensors"))
    ap.add_argument("--data", default=None, help="dataset dir (default characters/<id>/dataset)")
    ap.add_argument("--trigger", default=None, help="trigger word (default the character's lora_trigger or id)")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--dim", type=int, default=16, help="LoRA rank")
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--resolution", default="1024,1024")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    from shotforge.characters import load_character  # noqa: E402

    ch = load_character(args.character)
    trigger = (args.trigger or ch.lora_trigger or ch.id).strip()
    data_dir = args.data or os.path.join(HERE, "characters", ch.id, "dataset")
    if not os.path.isfile(args.base):
        raise SystemExit(f"[train] base checkpoint not found: {args.base}\n"
                         f"        run scripts/flux_setup.py (downloads Illustrious-XL), or pass --base.")

    ensure_sd_scripts()
    ensure_venv()
    root = build_dataset(ch.id, data_dir, trigger, args.repeats)

    out_dir = os.path.join("/content", f"lora_out_{ch.id}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        VENV_PY, "-m", "accelerate.commands.launch",
        "--num_processes", "1", "--num_machines", "1", "--mixed_precision", "bf16", "--dynamo_backend", "no",
        os.path.join(SD_SCRIPTS, "sdxl_train_network.py"),
        "--pretrained_model_name_or_path", args.base,
        "--train_data_dir", root,
        "--output_dir", out_dir, "--output_name", ch.id,
        "--resolution", args.resolution,
        "--network_module", "networks.lora",
        "--network_dim", str(args.dim), "--network_alpha", str(args.alpha),
        "--learning_rate", args.lr, "--text_encoder_lr", "5e-5",
        "--lr_scheduler", "cosine", "--lr_warmup_steps", "0",
        "--optimizer_type", "AdamW8bit",
        "--max_train_steps", str(args.steps),
        "--train_batch_size", str(args.batch),
        "--mixed_precision", "bf16", "--save_precision", "fp16",
        "--cache_latents", "--gradient_checkpointing", "--sdpa",
        "--enable_bucket", "--min_bucket_reso", "512", "--max_bucket_reso", "1536", "--bucket_reso_steps", "64",
        "--caption_extension", ".txt",
        "--save_model_as", "safetensors",
        "--max_data_loader_n_workers", "2", "--seed", "42",
    ]
    sh(*cmd, cwd=SD_SCRIPTS)

    produced = os.path.join(out_dir, f"{ch.id}.safetensors")
    if not os.path.isfile(produced):
        cands = glob.glob(os.path.join(out_dir, "*.safetensors"))
        if not cands:
            raise SystemExit(f"[train] training finished but no .safetensors in {out_dir}")
        produced = max(cands, key=os.path.getmtime)

    loras_dir = os.path.join(COMFY, "models", "loras")
    os.makedirs(loras_dir, exist_ok=True)
    comfy_lora = os.path.join(loras_dir, f"{ch.id}.safetensors")
    repo_lora = os.path.join(HERE, "characters", ch.id, f"{ch.id}.safetensors")
    shutil.copy(produced, comfy_lora)
    shutil.copy(produced, repo_lora)
    print(f"[ok] LoRA -> {comfy_lora}\n          {repo_lora} (back this up; Colab is ephemeral)")
    print(f"[next] set `lora: {ch.id}.safetensors` in characters/{ch.id}/character.yaml, then:\n"
          f"       python run.py frames-lora projects/<your-project>   (trigger word: {trigger!r})")


if __name__ == "__main__":
    main()
