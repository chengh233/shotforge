---
name: shotforge
description: >-
  Drive the shotforge image-to-video pipeline as an interactive loop: validate the
  approach on ONE shot first, auto-verify it, PAUSE for the user's decision, then full
  generation, then video. Use whenever the user wants to generate or iterate a
  shotforge project (frames → verify → video) — e.g. "用 shotforge 出 projects/X",
  "跑一下 freefall", "迭代这个项目的镜头".
---

# shotforge interactive loop

You orchestrate the shotforge pipeline. The GPU runs **remotely** (Colab ComfyUI via
`$COMFY_URL`); you run `python run.py <stage> ...` from the user's Mac — inputs upload
and outputs download automatically. Your job is to drive the commands AND pause at the
human-decision checkpoints so the user reviews and decides.

## Prompt strategy (your edge — read docs/PROMPTING.md)
The video model is weak; **compensate with detailed, structured prompts.** When you
author or refine a shot, EXPAND the user's brief into a full prompt:
- **Frame/still** (`content`/`frame_prompt`, → Qwen/FLUX): 美学(光线+景别+机位+构图+色调) +
  主体(具体外貌+神态) + 场景 + 风格. Be concretely specific (not "a girl" but her
  hair/eyes/outfit). Style comes from `style:` — don't repeat it.
- **Motion** (`prompt`/`action` + `camera`, → Wan I2V): the frame fixes the look, so write
  **运镜 + 一个清晰的主运动（节奏副词 + 强度 + 物理效果）**. One primary motion; pace adverbs
  (缓缓/迅速); physical effects (发丝飘动/热气/水花); NO contradictions; don't re-describe the
  static look; slow fast motion (cuts carry speed, not violent in-clip motion).
- Pull from the vocab palette in docs/PROMPTING.md (lighting / shot size / angle / lens /
  composition / camera move / motion verbs).
- Language: Wan motion = 中文 OK; Qwen still = 中文 OK; FLUX/FLUX-LoRA still = English better.
- When the user gives feedback, rewrite the shot's prompt richly per this, then regenerate.

## 0. Preconditions (check, don't assume)
- `$COMFY_URL` must point at the Colab cloudflared tunnel (else frames/video hit a dead
  local ComfyUI). If unset, ask the user for the tunnel URL (or to run `python run.py
  setup` on Colab first), then `export COMFY_URL=...`.
- `verify` needs `$GEMINI_API_KEY`. If missing, skip auto-verify and ask the user to
  review manually instead.
- Identify the project dir (ask if not given). Read its `project.yaml` to know the shots.

## 1. Minimal effort FIRST — never generate all shots blindly
Validate the prompt/approach on ONE representative shot (default the first; ask if
unsure which):
1. `python run.py frames <project> --shot <id> --overwrite`
2. `python run.py verify <project> --shot <id>`   (skip if no GEMINI key)
3. Report to the user: the frame path (`<project>/frames/<id>.jpeg`) and the verify
   output (score / 问题 / 建议). Tell them to look at the image.
4. **PAUSE.** Use AskUserQuestion: 通过(全量) / 改提示词重试这一镜 / 换一镜验证 / 停。
   Wait for their decision. If they give prompt feedback, edit `project.yaml` yourself,
   then re-run step 1–3 for that shot.

## 2. Full generation (only after the approach is approved)
1. `python run.py frames <project>`  (all shots)
2. `python run.py verify <project>`  — present the report; list shots scoring below
   threshold and their 建议.
3. **PAUSE.** Ask the user which flagged shots to regenerate and whether to tweak their
   prompts. Apply the edits to `project.yaml`, then `frames --shot <id> --overwrite` +
   `verify --shot <id>` on those. Loop until the user is satisfied.
4. Optional: `python run.py manga <project>` so they can read it as a storyboard.

## 3. Video + finish (only after the user approves the frames)
1. `python run.py video <project>`   (remote GPU; downloads mp4s locally)
2. `python run.py dub <project>` → `python run.py subs <project>` → `python run.py post
   <project> [--music ..] [--crossfade ..]`  (all local/CPU)
3. Report the final mp4 path; PAUSE and ask whether to iterate (back to step 1/2) or stop.

## Rules
- ALWAYS pause and wait after a `verify` (or after one-shot generation) — never
  auto-advance to full generation or video. The user's "go" gates each escalation.
- Cost/time order is sacred: **one shot → verify → approve → full → approve → video.**
- Report faithfully: show real `verify` scores and failures; if a command errors, paste
  the error and stop (don't pretend it worked).
- When the user gives prompt/scene/character feedback, edit `project.yaml` (or the
  relevant `characters/ scenes/ styles/` file) yourself, then regenerate only the
  affected shots.
- Commit/push only when the user explicitly asks.
- For decoupling/engine/consistency questions, the design lives in the repo
  (`shotforge/`, `CLAUDE.md`) and the architecture memory.
