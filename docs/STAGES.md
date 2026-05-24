# 分阶段流水线：在哪做、怎么单独跑、怎么省 GPU

完整短剧 = 剧本 → 起始帧 → **视频(I2V)** → 配音 → 字幕 → 配乐 → 后期合成。
每一步都能单独跑，方便你在进入下一步前检查质量。

核心原则（省 GPU 钱 + 让 GPU 用在刀刃上）：

> **Colab GPU 只做一件事：视频生成(I2V)。** 其余全部在 MacBook（CPU）上做。
> GPU 只在批量渲染时开、渲完就断；CPU 的活（字幕/配音/拼接/后期）不为它付 GPU 钱。

## 各阶段该在哪做

| 阶段 | 在哪 | 要 GPU? | 说明 |
|---|---|---|---|
| 剧本 / 分镜 | **Mac**（Claude Code） | 否 | 纯文本，写 `project.yaml` |
| 起始帧（文生图） | **Mac：即梦/Dreamina（推荐）** 或本地 SDXL | 是(本地开源) | 要「先看图再渲视频」的质量门；放 Mac/在线 → 不占 Colab GPU。即梦免费、动漫+中文强、零 setup。也可在 Colab 的 ComfyUI 里出图（很便宜，但会占着 GPU 等你 review） |
| **视频 I2V** | **Colab GPU** | **是，重** | 唯一真正吃 GPU 的活，批量跑满 |
| 配音 TTS | **Mac** | 否 | `tools.dub` 用 edge-tts，在线、无 GPU |
| 字幕 | **Mac** | 否 | `tools.subtitle`，纯文本 + ffmpeg |
| 配乐 | **Mac** | 否/轻 | 免版税音乐，或轻量 MusicGen（CPU/MPS）；不必占 GPU |
| 拼接 / 后期合成 | **Mac** | 否 | `tools.post`，ffmpeg(CPU) |
| 查看产物 | **Mac** | 否 | 别让 Colab 开着 GPU 空转等你看片 |

**为什么这样分**：视频生成是唯一计算量大、必须 GPU 的环节；其它要么是文本/ffmpeg(CPU)、要么有零-GPU 的在线方案。把它们全留在 Mac，Colab 的 GPU 时间就只花在渲染上。

### 进阶：Mac 远程驱动 Colab 的 GPU（GPU 空转最小）
让 Colab **只跑 ComfyUI 当渲染服务器**，shotforge 在 **Mac** 上用隧道地址远程调它：
```bash
# Colab：起服务 + 隧道，拿到 https://xxx.trycloudflare.com
# Mac：
python -m shotforge.generate --project projects/example --engine comfy \
  --comfy-url https://xxx.trycloudflare.com
```
起始帧从 Mac 上传、成片自动下载回 Mac。编排/IO/后期都在 Mac，**GPU 只在真正算的时候转**。

## 从空白 Colab 一键起步

```bash
# Colab cell（全新机器）：克隆 + 一键 setup（自动检查、缺什么下什么、起服务）
!git clone https://github.com/chengh233/shotforge /content/shotforge 2>/dev/null; \
 cd /content/shotforge && git pull -q && python scripts/colab_setup.py
```
Mac 侧只需轻量依赖（无 torch）：`pip install pyyaml requests edge-tts`（ffmpeg 自带）。

## 分阶段命令（每步可单独跑 + 看产物）

每个阶段也有统一入口 `python run.py <stage> <project> [args]`。

### ① 视频 I2V（Colab GPU）
```bash
python run.py serve                                  # 起 ComfyUI（等 "up on :8188"）
python run.py video projects/example --shot s1       # 先单镜验证
python run.py video projects/example                 # 全部镜头
```
**看产物**（Colab cell 里）：
```python
from IPython.display import Video
Video("/content/shotforge/projects/example/out/s1.mp4", embed=True, width=300)
```

### ② 配音（Mac）
```bash
python run.py dub projects/example --voice zh-CN-XiaoxiaoNeural    # 每句台词 → out/audio/<id>.mp3
```
**看产物**：`afplay projects/example/out/audio/s1.mp3`（Mac 播放）

### ③ 字幕（Mac）
```bash
python run.py subs projects/example                  # → out/<片名>.srt（按各镜时长对齐）
```
**看产物**：`cat projects/example/out/*.srt`

### ④ 配乐（Mac，可选）
用 AI 生成或免版税，拿到一个 `bgm.mp3`（见下「模型推荐」）。

### ⑤ 后期合成（Mac）
```bash
python run.py post projects/example                          # 拼接 + 配音 + 烧字幕（硬切）
python run.py post projects/example --crossfade 0.5 --fade 0.6  # 镜头间溶解 + 首尾淡入淡出
python run.py post projects/example --music bgm.mp3          # 再加背景音乐（自动压低音量混音）
```
**看产物**：`open projects/example/out/夏风_final.mp4`（Mac 打开）

> 只想要无声拼接：`python run.py stitch projects/example` → `out/<片名>_full.mp4`。

## 常用查看产物命令速查

```bash
ls -lh projects/example/out projects/example/out/audio    # 看都生成了啥 + 大小
ffprobe -v error -show_entries format=duration -of csv=p=0 projects/example/out/s1.mp4   # 时长
open  projects/example/out/夏风_final.mp4                  # Mac 打开视频
afplay projects/example/out/audio/s1.mp3                   # Mac 播放音频
cat   projects/example/out/夏风.srt                        # 看字幕
```
（Colab 里看视频/图片用 `IPython.display` 的 `Video` / `Image`。）

## AI 模型推荐

**配音 TTS**
- **edge-tts**（默认，`tools.dub` 用）：在线、免费、**无 GPU**、中文好。先用它。
- 想要更自然 / 声音克隆（本地、要点 GPU）：**CosyVoice2**（阿里，中文最强之一）、**IndexTTS2**、**F5-TTS**。

**配乐**
- **MusicGen**（Meta）/ **Stable Audio Open**：开源音乐生成，给「calm nostalgic anime piano, summer」这类提示词出 BGM。MusicGen-small 可在 CPU/MPS 慢跑，或 Colab 上顺手生成一次。
- 或直接用**免版税音乐**（零计算）。
- 生成/拿到 `bgm.mp3` 后用 `tools.post --music bgm.mp3` 混进去。

**起始帧文生图**：见 `docs/COMFYUI.md`（Animagine XL / Illustrious / Qwen-Image），或用即梦/Dreamina。

## 省 GPU、又把 GPU 用满的策略

- **只渲染上 GPU**；字幕/配音/拼接/后期/review 全在 Mac → 不为 CPU 活付 GPU 钱。
- **批量渲染**：一个 session 里把所有镜头（甚至整集）连着渲完，把「开机+下模型」的固定成本摊薄到尽量多的镜头上。
- **下载用 aria2 快下**（`colab_setup.py` 已用），缩短「GPU 空转等下载」的窗口。
- **review 在 Mac 做**：渲完下载 clips 再看，别让 Colab 开着 GPU 等你看片。
- **高端 GPU（A100/H100）**：每秒更快 + 能上更高分辨率/步数 → 同样 GPU-time 出更高质量。想「充分利用」就：批量 + 拉高分辨率/步数 + 不留交互空隙。
- **进阶**：Mac 远程驱动（上面那节）——Colab 纯 GPU 渲染，空转最小。
