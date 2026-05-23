# shotforge

A minimal, script-agnostic **image-to-video pipeline for short dramas (短剧)**.

You author scripts and prepare starting-frame images locally (e.g. on a Mac),
commit, and push. A GPU box (Google Colab, single **L4 24GB**) pulls the repo
and renders every shot to mp4. The default video model is **LTX-Video** via the
`diffusers` library.

**Design goal: one renderer, many projects.** A new script is just a new folder
under `projects/`.

## Layout

```
shotforge/        # the renderer (model-agnostic image-to-video)
  manifest.py     # Shot/Project dataclasses + project.yaml loader
  i2v.py          # diffusers pipeline wrapper (LTX-Video by default)
  generate.py     # CLI: render a project's shots to mp4
tools/
  stitch.py       # concat per-shot mp4s into one film (ffmpeg)
  last_frame.py   # export a clip's last frame (for chaining longer takes)
projects/
  example/        # a project = ONE script
    project.yaml  #   shot list + shared defaults
    frames/       #   starting-frame PNGs: s1.png, s2.png, ...
    out/          #   rendered mp4s (gitignored)
```

## Flow

**Local (Mac):**

1. Copy a project folder for your new script:
   `cp -r projects/example projects/ep01`
2. Edit `projects/ep01/project.yaml` — shot ids, prompts, lengths.
3. Drop your starting frames into `projects/ep01/frames/` as `s1.png`,
   `s2.png`, … (you can export these from **Dreamina** at 9:16).
4. Commit and push.

**GPU box (Colab L4):**

```bash
git pull
bash setup.sh                                       # installs deps (NOT torch)
python -m shotforge.generate --project projects/ep01
python -m tools.stitch       --project projects/ep01   # optional: one film
```

While iterating, render a single shot with `--shot s2`.

## "Project = one script"

Everything about an episode lives in its `projects/<name>/` folder — the
manifest, the frames, and (after rendering) the output clips. The renderer code
never changes per project. To make a new drama, copy the folder and edit it.

## Going past 5 seconds

LTX clips are short by nature. Three ways to get longer content:

1. **Per-shot `seconds`** — bump a shot's `seconds:` in `project.yaml`
   (e.g. `seconds: 8`). The frame count is snapped to LTX's required `8*N + 1`.
2. **Many short shots + stitch** — write several shots, then concatenate them
   into one continuous film with `tools.stitch`.
3. **`last_frame` chaining** — render a shot, extract its final frame, and use
   that PNG as the next sub-shot's starting frame for a seamless longer take:
   ```bash
   python -m tools.last_frame --video projects/ep01/out/s1.mp4 \
                              --out   projects/ep01/frames/s1b.png
   ```

## Swapping the model (e.g. Wan 2.2)

The pipeline class and weights are configurable:

- Point the loader at other weights:
  `export I2V_MODEL_ID="Wan-AI/Wan2.2-I2V-A14B-Diffusers"`
- In `shotforge/i2v.py`, swap `LTXImageToVideoPipeline` for
  `WanImageToVideoPipeline` (also from `diffusers`). The call signature
  (`image`, `prompt`, `width`, `height`, `num_frames`, …) is the same.

Note: the `8*N + 1` frame rule is **LTX-specific**; other models may have
different constraints — see `frames_for()` in `shotforge/manifest.py`.
