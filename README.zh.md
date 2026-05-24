# shotforge

[English](./README.md) · **简体中文**

一个极简的**短剧（图生视频）流水线**——在笔记本上写剧本、备起始帧，在 GPU 机器上把
每个镜头渲染成视频，再加配音、字幕、配乐。全程命令行驱动。

- **Mac 创作，GPU 机器（如 Colab）渲染。** 提交并推送剧本和帧；GPU 机器拉取并渲染。
- **渲染：ComfyUI + Wan 2.2 图生视频**——高质量路径。也保留了本地 `diffusers` 引擎，
  但 diffusers 的 Wan I2V 不稳（会融化），请用 ComfyUI。见 [`docs/COMFYUI.md`](./docs/COMFYUI.md)。
- **GPU 只做视频。** 起始帧、配音、字幕、配乐、最终合成都在 Mac（CPU）上做，把 GPU
  时间降到最低。见 [`docs/STAGES.md`](./docs/STAGES.md)。
- **一套渲染器，多个项目**——新剧本只是 `projects/` 下的一个新文件夹。

## 流水线（每步可单独跑，便于逐步检查质量）

```
剧本/分镜 → 起始帧 → 视频(I2V) → 配音 → 字幕 → 配乐 → 成片
   Mac      Mac/即梦   Colab GPU    Mac     Mac    Mac    Mac
```

## 快速开始

**全新 Colab GPU 机器**（一个 cell——装 ComfyUI + Wan 模型并起服务）：
```bash
!git clone https://github.com/chengh233/shotforge /content/shotforge 2>/dev/null; \
 cd /content/shotforge && git pull -q && python scripts/colab_setup.py
```

**渲染 + 收尾**——分阶段，`python run.py <阶段> <项目>`：
```bash
python run.py video  projects/example --shot s1   # 先单镜验证   (Colab GPU)
python run.py video  projects/example             # 全部镜头     (Colab GPU)
python run.py dub    projects/example             # 配音         (Mac, edge-tts)
python run.py subs   projects/example             # 字幕         (Mac)
python run.py post   projects/example             # 拼接 + 配音 + 字幕 -> 成片 (Mac)
python run.py post   projects/example --music bgm.mp3   # 再加背景音乐
```

每步在哪做、怎么看产物的详细命令：[`docs/STAGES.md`](./docs/STAGES.md)。
一段视频是怎么生成的 + 调试：[`docs/PIPELINE.md`](./docs/PIPELINE.md)。

## 目录结构

```
shotforge/
  manifest.py     # Shot/Project 数据类 + project.yaml 加载器
  backends.py     # 模型注册表（Wan / LTX）：每个模型的帧数与尺寸规则
  comfy.py        # comfy 引擎：通过 HTTP 驱动 ComfyUI 服务   ← 推荐
  i2v.py          # diffusers 引擎（本地管线；LTX 可用，Wan I2V 会融化）
  generate.py     # 命令行：渲染镜头  (--engine comfy | diffusers)
tools/
  stitch.py       # 把各镜头拼成一部无声片
  last_frame.py   # 导出片段最后一帧（衔接更长镜头）
  dub.py          # 配音（TTS）：用每镜的 dialogue 生成        (CPU)
  subtitle.py     # 按片段时长对齐生成 SRT                     (CPU)
  post.py         # 最终合成：拼接 + 配音 + 配乐 + 烧字幕       (CPU)
run.py            # 分阶段入口
scripts/          # colab_setup.py —— 装 ComfyUI + 下 Wan 模型 + 起服务 + 验证
comfyui/          # wan_i2v_api.json —— comfy.py 驱动的 ComfyUI 工作流
docs/             # PIPELINE.md(原理/调试)、COMFYUI.md(渲染搭建)、STAGES.md(分阶段)
projects/
  example/        # 一个项目 = 一个剧本：project.yaml + frames/ + out/
```

## “一个项目 = 一个剧本”

一集的所有内容都在 `projects/<name>/`：`project.yaml`（镜头列表——每个镜头有起始帧、
运动提示词、可选的 `dialogue` 台词）、`frames/` 起始帧、`out/` 渲染产物。渲染器代码不
随项目改变；做新剧，复制文件夹再改即可。

## 模型与引擎

- **`--engine comfy`**（推荐）——驱动跑着 **Wan 2.2 I2V** 的 ComfyUI，是能出高质量的
  图生视频路径；umT5 文本编码器支持中文提示词。搭建见 [`docs/COMFYUI.md`](./docs/COMFYUI.md)。
- **`--engine diffusers`**（默认值）——本地 diffusers 管线；LTX-Video 可用，但 Wan I2V
  在 diffusers 里会飘/融化，这正是 comfy 引擎存在的原因。模型注册表：`shotforge/backends.py`。
- 想让视频连贯、不像 AI（动漫风 + 细微动作 + 干净起始帧），见 [`docs/PIPELINE.md`](./docs/PIPELINE.md) 的经验。
```
