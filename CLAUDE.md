# shotforge — context for Claude Code

**What this is:** a minimal, script-agnostic image-to-video pipeline for making
short dramas (短剧). Author scripts + starting-frame PNGs locally, push to git;
a GPU box (Colab, single L4 24GB) pulls and renders each shot to mp4 with a
diffusers I2V model (Wan 2.2 by default, LTX-Video also supported). **One
renderer, many projects** — a new script is a new folder under `projects/`.

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
  (default `wan`). Known backends: `wan` (Wan 2.2, default — best for Chinese
  prompts) and `ltx` (LTX-Video). Add one by appending to `BACKENDS` in
  `backends.py`.
- LTX's text encoder is T5 (English-centric) → write motion prompts in English.
  Wan's is umT5 (multilingual) → Chinese prompts work. The drama's dialogue is
  unrelated: these models output **silent video, no subtitles** (post-process
  separately).
- Wan defaults to the 5B TI2V variant; the A14B (MoE) variant won't fit a 24GB
  L4. The Wan repo id / class / frame rules are best-effort — verify on the GPU
  box (load_pipe prints available classes if the name is wrong).

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
- **OOM on the L4 (24GB)** — lower `width`/`height` and/or `seconds`, and
  confirm `pipe.enable_model_cpu_offload()` actually ran (the cuda branch in
  `load_pipe`). VAE tiling is enabled best-effort.
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
