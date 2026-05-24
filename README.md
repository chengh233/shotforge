# shotforge

**English** · [简体中文](./README.zh.md)

A minimal pipeline for making **short dramas (短剧) from images** — author the
script and starting frames on your laptop, render each shot to video on a GPU
box, then add voiceover, subtitles and music. All driven from the command line.

- **Author on your Mac, render on a GPU box (e.g. Colab).** Commit + push the
  script and frames; the GPU box pulls and renders.
- **Rendering: ComfyUI + Wan 2.2 image-to-video** — the quality path. A local
  `diffusers` engine also exists, but diffusers' Wan I2V is unreliable (melts);
  use ComfyUI. See [`docs/COMFYUI.md`](./docs/COMFYUI.md).
- **GPU does only video.** Frames, voiceover, subtitles, music and final
  assembly run on the Mac (CPU) to keep GPU time minimal. See
  [`docs/STAGES.md`](./docs/STAGES.md).
- **One renderer, many projects** — a new drama is a new folder under `projects/`.

## Pipeline (each stage runs on its own; check quality between steps)

```
script/storyboard → starting frames → video (I2V) → voiceover → subtitles → music → final
     Mac               Mac / 即梦       Colab GPU       Mac         Mac        Mac     Mac
```

## Quickstart

**Fresh Colab GPU box** (one cell — installs ComfyUI + Wan models, starts the server):
```bash
!git clone https://github.com/chengh233/shotforge /content/shotforge 2>/dev/null; \
 cd /content/shotforge && git pull -q && python scripts/colab_setup.py
```

**Render + finish** — staged, `python run.py <stage> <project>`:
```bash
python run.py video  projects/example --shot s1   # one shot first   (Colab GPU)
python run.py video  projects/example             # all shots        (Colab GPU)
python run.py dub    projects/example             # voiceover        (Mac, edge-tts)
python run.py subs   projects/example             # subtitles        (Mac)
python run.py post   projects/example             # concat + VO + subs -> final (Mac)
python run.py post   projects/example --music bgm.mp3   # + background music
```

Step-by-step with where each stage runs and how to view each artifact:
[`docs/STAGES.md`](./docs/STAGES.md). How a clip is generated + debugging:
[`docs/PIPELINE.md`](./docs/PIPELINE.md).

## Layout

```
shotforge/
  manifest.py     # Shot/Project dataclasses + project.yaml loader
  backends.py     # model registry (Wan / LTX): frame & dimension rules per model
  comfy.py        # comfy engine: drive a ComfyUI server over HTTP   ← recommended
  i2v.py          # diffusers engine (local pipeline; LTX ok, Wan I2V melts)
  generate.py     # CLI: render shots  (--engine comfy | diffusers)
tools/
  stitch.py       # concat clips into one silent film
  last_frame.py   # export a clip's last frame (chain longer takes)
  dub.py          # voiceover (TTS) from each shot's dialogue        (CPU)
  subtitle.py     # SRT timed to clip lengths                        (CPU)
  post.py         # final mux: clips + voiceover + music + subtitles (CPU)
run.py            # staged runner (python run.py <stage> <project>)
scripts/          # colab_setup.py — install ComfyUI + Wan models, serve, verify
comfyui/          # wan_i2v_api.json — the ComfyUI workflow comfy.py drives
docs/             # PIPELINE.md (how/debug), COMFYUI.md (render setup), STAGES.md (stages)
projects/
  example/        # a project = ONE script: project.yaml + frames/ + out/
```

## A project = one script

Everything about an episode lives in `projects/<name>/`: the `project.yaml`
(shot list — each shot has a starting frame, a motion prompt, and optional
`dialogue`), the `frames/`, and the rendered `out/`. The renderer never changes
per project; to make a new drama, copy the folder and edit it.

## Models & engines

- **`--engine comfy`** (recommended) — drives ComfyUI running **Wan 2.2 I2V**,
  the working high-quality image-to-video path; umT5 handles Chinese prompts.
  Setup: [`docs/COMFYUI.md`](./docs/COMFYUI.md).
- **`--engine diffusers`** (default flag value) — local diffusers pipeline; fine
  for LTX-Video, but Wan I2V drifts/melts in diffusers — which is why the
  ComfyUI engine exists. Model registry: `shotforge/backends.py`.
- For coherent, non-AI-looking video (anime style + subtle motion + clean
  frames), see the guidance in [`docs/PIPELINE.md`](./docs/PIPELINE.md).
