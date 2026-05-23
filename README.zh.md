# shotforge

[English](./README.md) · **简体中文**

一个极简、与剧本无关的**短剧图生视频（image-to-video）流水线**。

你在本地（比如 Mac 上）撰写剧本、准备每个镜头的起始帧图片，提交并推送到 git；一台
GPU 机器（Google Colab，单张 **L4 24GB**）拉取仓库，把每个镜头渲染成 mp4。默认的
视频模型是通过 `diffusers` 库调用的 **LTX-Video**。

**设计目标：一套渲染器，多个项目。** 新剧本只是 `projects/` 下的一个新文件夹。

## 目录结构

```
shotforge/        # 渲染器（与具体模型无关的图生视频）
  manifest.py     # Shot/Project 数据类 + project.yaml 加载器
  i2v.py          # diffusers 流水线封装（默认 LTX-Video）
  generate.py     # 命令行：把一个项目的所有镜头渲染成 mp4
tools/
  stitch.py       # 用 ffmpeg 把各镜头的 mp4 拼接成一整部片子
  last_frame.py   # 导出某个片段的最后一帧（用于衔接更长的镜头）
projects/
  example/        # 一个项目 = 一个剧本
    project.yaml  #   镜头列表 + 共享默认值
    frames/       #   起始帧 PNG：s1.png、s2.png …
    out/          #   渲染输出的 mp4（已被 git 忽略）
```

## 工作流程

**本地（Mac）：**

1. 为新剧本复制一份项目文件夹：
   `cp -r projects/example projects/ep01`
2. 编辑 `projects/ep01/project.yaml`——镜头 id、提示词、时长。
3. 把起始帧放进 `projects/ep01/frames/`，命名为 `s1.png`、`s2.png` ……
   （可以用 **Dreamina（即梦）** 按 9:16 导出这些图片。）
4. 提交并推送。

**GPU 机器（Colab L4）：**

```bash
git pull
bash setup.sh                                       # 安装依赖（不含 torch）
python -m shotforge.generate --project projects/ep01
python -m tools.stitch       --project projects/ep01   # 可选：拼成一整部
```

调试时可以用 `--shot s2` 只渲染单个镜头。

## “一个项目 = 一个剧本”

一集的所有内容都放在它自己的 `projects/<name>/` 文件夹里——清单（manifest）、起始
帧，以及渲染后的输出片段。渲染器代码不会因项目而改变。要做新剧，复制文件夹再改内容
即可。

## 如何超过 5 秒

LTX 的片段天生较短。三种获得更长内容的方式：

1. **逐镜头设置 `seconds`**——在 `project.yaml` 里调高某个镜头的 `seconds:`
   （例如 `seconds: 8`）。帧数会被自动对齐到 LTX 要求的 `8*N + 1`。
2. **多个短镜头 + 拼接**——写多个镜头，再用 `tools.stitch` 拼成一整部连续的片子。
3. **`last_frame` 衔接**——渲染一个镜头，导出它的最后一帧，把这张 PNG 作为下一个
   子镜头的起始帧，从而得到无缝衔接的更长镜头：
   ```bash
   python -m tools.last_frame --video projects/ep01/out/s1.mp4 \
                              --out   projects/ep01/frames/s1b.png
   ```

## 更换模型（例如 Wan 2.2）

流水线类和权重都是可配置的：

- 指向其他权重：
  `export I2V_MODEL_ID="Wan-AI/Wan2.2-I2V-A14B-Diffusers"`
- 在 `shotforge/i2v.py` 里，把 `LTXImageToVideoPipeline` 换成
  `WanImageToVideoPipeline`（同样来自 `diffusers`）。调用签名（`image`、`prompt`、
  `width`、`height`、`num_frames` 等）保持不变。

注意：`8*N + 1` 这条帧数规则是 **LTX 特有的**；其他模型可能有不同的约束——参见
`shotforge/manifest.py` 里的 `frames_for()`。
