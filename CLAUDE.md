# shotforge — context for Claude Code

**What this is:** a minimal, script-agnostic image-to-video pipeline for making
short dramas (短剧). Author scripts + starting-frame PNGs locally, push to git;
a GPU box (Colab, single L4 24GB) pulls and renders each shot to mp4 with
LTX-Video via `diffusers`. **One renderer, many projects** — a new script is a
new folder under `projects/`.

## Layout
- `shotforge/manifest.py` — `Shot`/`Project` dataclasses, `load_project`,
  `frames_for`, `snap32`.
- `shotforge/i2v.py` — diffusers pipeline wrapper; `MODEL_ID` from
  `$I2V_MODEL_ID` (default `Lightricks/LTX-Video`); lazy singleton pipe.
- `shotforge/generate.py` — CLI render loop (`python -m shotforge.generate`).
- `tools/stitch.py` — ffmpeg concat of per-shot mp4s.
- `tools/last_frame.py` — export a clip's last frame for chaining.
- `projects/<name>/` — one script: `project.yaml` + `frames/` + `out/`.

## Run
```bash
bash setup.sh                                              # deps, NOT torch
python -m shotforge.generate --project projects/example
python -m shotforge.generate --project projects/example --shot s2   # one shot
python -m tools.stitch       --project projects/example
python -m tools.last_frame   --video <clip>.mp4 --out <next>.png
```

## Common breakage points
- **`num_frames` must be `8*N + 1`** and **width/height divisible by 32**.
  Handled by `frames_for()` and `snap32()` in `manifest.py` — don't bypass them
  or LTX will error / produce garbage.
- **OOM on the L4 (24GB)** — lower `width`/`height` and/or `seconds`, and
  confirm `pipe.enable_model_cpu_offload()` actually ran (the cuda branch in
  `load_pipe`). VAE tiling is enabled best-effort.
- **Frame not found** — frame paths in `project.yaml` are relative to the
  project dir; `load_project` joins them. Check `--project` points at the right
  folder.
- **mps Generator unsupported** — on Apple Silicon, `torch.Generator` is created
  on `cpu` (see `generate` in `i2v.py`); don't move it to mps.
- **diffusers class names drift** — pipeline class names occasionally change
  between `diffusers` versions; if an import fails, check the installed
  version's class name (e.g. LTX vs Wan I2V pipeline).

## Conventions
- Python 3.10+, type hints, small focused modules, no heavy frameworks.
- Clear stdout logging with `[tag]` prefixes.
- **Commit `frames/*.png`. Never commit `out/` or `*.mp4`** (see `.gitignore`).
