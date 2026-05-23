# shotforge — context for Claude Code

**What this is:** a minimal, script-agnostic image-to-video pipeline for making
short dramas (短剧). Author scripts + starting-frame PNGs locally, push to git;
a GPU box (Colab; 24GB works via offload, 40GB+ runs the 14B resident) pulls and
renders each shot to mp4 with a diffusers I2V model (Wan2.1-I2V-14B by default,
LTX-Video also supported). **One renderer, many projects** — a new script is a
new folder under `projects/`.

## Layout
- `shotforge/backends.py` — `Backend` dataclass + `BACKENDS` registry +
  `get_backend`. Each model carries its own frame/dim rules and defaults.
- `shotforge/manifest.py` — `Shot`/`Project` dataclasses, `load_project`,
  `frames_for`, `snap_dim`.
- `shotforge/i2v.py` — diffusers pipeline wrapper; resolves the backend's
  pipeline class lazily; `$I2V_MODEL_ID` overrides the checkpoint; singleton
  pipe keyed by (class, checkpoint).
- `shotforge/generate.py` — CLI render loop (`python -m shotforge.generate`).
- `tools/stitch.py` — ffmpeg concat of per-shot mp4s.
- `tools/last_frame.py` — export a clip's last frame for chaining.
- `projects/<name>/` — one script: `project.yaml` + `frames/` + `out/`.
- `docs/PIPELINE.md` — how a video is generated end-to-end + a debug guide.

## Models
- A project picks its model with the top-level `model:` field in `project.yaml`
  (default `wan`). Backends in `backends.py`:
  - `wan` — Wan2.1-I2V-14B-**720P** (default, best for Chinese dramas). Proper
    image-to-video: `WanImageToVideoPipeline` auto-loads the CLIP image_encoder.
  - `wan-480p` — same family at 480P, lighter/faster for draft renders.
  - `ltx` — LTX-Video, light/fast but English-centric (T5).
- **I2V = image + text**, not image alone: the starting frame fixes the look,
  the per-shot `prompt` drives the motion. Both are required.
- Wan's text encoder is umT5 (multilingual) → Chinese motion prompts work; LTX's
  T5 wants English. These models output **silent video, no subtitles**
  (post-process separately — see `docs/PIPELINE.md`).
- Wan specifics handled in `load_pipe`: VAE + image_encoder forced to fp32, UniPC
  `flow_shift` (5.0@720P / 3.0@480P), 16fps (the trained rate). 14B bf16 (~28GB)
  wants a 40GB+ GPU resident; smaller cards auto cpu-offload (`$I2V_OFFLOAD`).
- Speed: 720P 14B is heavy (~75k attention tokens). For iteration use
  `model: wan-480p` + fewer `steps`. Knobs: `$I2V_COMPILE=1` (or `=max-autotune`)
  `torch.compile`s the transformer (~1.5-2x after a slow first run);
  `$I2V_ATTN=flash` forces the fused attention kernels (and proves whether
  attention was silently using the slow math fallback). `load_pipe` prints the
  enabled SDP backends.
- Class drift: `load_pipe` prints the installed diffusers' I2V class names if a
  backend's `pipeline_cls` is wrong.

## Run
```bash
bash setup.sh                                              # deps, NOT torch
python -m shotforge.generate --project projects/example
python -m shotforge.generate --project projects/example --shot s2   # one shot
python -m tools.stitch       --project projects/example
python -m tools.last_frame   --video <clip>.mp4 --out <next>.png
```

## Common breakage points
- **`num_frames` must be `quantum*N + 1`** (LTX `8`, Wan `4`) and
  **width/height divisible by the backend's `dim_multiple`** (LTX `32`, Wan
  `16`). Handled by `frames_for()` and `snap_dim()` in `manifest.py` using the
  project's backend — don't bypass them or the model will error / make garbage.
- **OOM / model too big** — 14B Wan in bf16 is ~28GB and won't fit a 24GB card
  resident. `load_pipe` keeps the model resident on >=40GB GPUs and cpu-offloads
  below that automatically (force with `$I2V_OFFLOAD=1`/`0`). Also lower
  `width`/`height`/`seconds`; VAE runs fp32 (`$I2V_VAE_DTYPE=bf16` to halve it),
  and `$I2V_VAE_TILING=1` tiles the decode.
- **Frames go blank after the first** — most often the WRONG model for the I2V
  pipeline: `WanImageToVideoPipeline` needs a real I2V checkpoint with a CLIP
  `image_encoder` (Wan2.1-I2V-14B-480P/720P). Wan2.2-TI2V-5B has no image_encoder
  and its I2V is broken in diffusers — don't use it here. Other causes: VAE NaN
  in low precision (fp32 is the default) or VAE tiling (off by default).
- **Frame not found** — frame paths in `project.yaml` are relative to the
  project dir; `load_project` joins them. Check `--project` points at the right
  folder.
- **mps Generator unsupported** — on Apple Silicon, `torch.Generator` is created
  on `cpu` (see `generate` in `i2v.py`); don't move it to mps.
- **diffusers class names drift** — pipeline class names occasionally change
  between `diffusers` versions. `load_pipe` resolves the class by name and, if
  it's missing, raises with the list of I2V classes the installed version
  exposes; update the offending backend's `pipeline_cls` in `backends.py`.

## Conventions
- Python 3.10+, type hints, small focused modules, no heavy frameworks.
- Clear stdout logging with `[tag]` prefixes.
- **Commit `frames/*.png`. Never commit `out/` or `*.mp4`** (see `.gitignore`).
