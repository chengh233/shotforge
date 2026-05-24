# ComfyUI 渲染后端（方案 B）

diffusers 的 Wan 图生视频出图融化/崩坏（我们试了 fp32 VAE、CLIPVisionModel、auto-resolution、升级 diffusers 都没解决——这是 diffusers Wan I2V 实现本身的已知质量问题）。**ComfyUI 用的是另一套、社区公认更好的 Wan 实现**，是高质量 Wan 视频的主流路径。

方案 B = **shotforge 继续做编排（剧本/批量/拼接），渲染换成调 ComfyUI**。

---

## TL;DR：完整操作步骤（端到端，全部在 Colab 上跑）

```bash
# A) 一次性安装（装过可跳过）
cd /content/shotforge && git pull
python scripts/colab_setup.py        # ComfyUI + Wan 模型
pip install requests

# B) 后台启动 ComfyUI 服务，并等它就绪
python scripts/colab_setup.py        # 打印 "ComfyUI up on :8188" 才算好

# C) 确认 B 起来后，用 shotforge 调它渲染
python -m shotforge.generate --project projects/example --engine comfy --shot s1
# 全部镜头 + 拼接：
python -m shotforge.generate --project projects/example --engine comfy
python -m tools.stitch       --project projects/example
```

**最常见错误 `Connection refused (127.0.0.1:8188)` = ComfyUI 服务没在跑。** 先跑 B、
看到 `ComfyUI up on :8188` 再跑 C。手动检查：
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/   # 200 = 好
tail -n 30 /content/comfyui.log                                   # 起不来就看日志
```

> 纯自动渲染**不需要** cloudflared 隧道；隧道只在你想从 Mac 浏览器看 GUI / 手动调工作流时才用（见下文 Phase 1）。

---

## 生成起始帧（文生图，也用 ComfyUI）

ComfyUI 也能做文生图，所以 **T2I + I2V 一个工具搞定**：生成分镜图 → GUI 里预览 →
满意就存 → 作为 I2V 的起始帧。

**模型推荐（动漫）**：
- **Animagine XL 4.0** 或 **Illustrious 系**（如 Nova Anime XL）：SDXL 动漫专精，约
  6.5GB，A100 秒出，**下载小**（新会话重下也快）——做动漫首选。
- **Qwen-Image**：当前最强开源通用模型、原生支持 ComfyUI、中文文字渲染好；但模型大、
  显存/下载重，想要顶级质量或画面里要写中文字时再用。

**装一个动漫 SDXL（以 Animagine XL 4.0 为例，用 hf_transfer 加速）**：
```bash
pip install -q hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download cagliostrolab/animagine-xl-4.0 animagine-xl-4.0.safetensors \
  --local-dir /content/ComfyUI/models/checkpoints
# 文件名若不符，去该模型 HF 页的 Files 标签确认；装完重启 ComfyUI 才扫得到
```

**用法**：ComfyUI 里 Browse Templates → Image → SDXL txt2img 模板 → 选这个 checkpoint
→ 写动漫提示词 → Queue → 右侧预览 → 满意就 Save，作为 I2V 起始帧（放进 `frames/`）。

---

## 一键生成一致性起始帧（Flux Kontext · `frames` 阶段）

`frames` 阶段用 **Flux Kontext**：你给 **1 张角色参考图**，它让每个分镜都「同一个人换场景」，
自动保持一致——不用一张张手动出图。

**1. 装 Kontext 模型**（GPU box，装完会自动重启 ComfyUI）：
```bash
python scripts/flux_setup.py
```
| 文件 | 目录 |
|---|---|
| `flux1-dev-kontext_fp8_scaled.safetensors` | `diffusion_models/` |
| `ae.safetensors` | `vae/` |
| `clip_l.safetensors` | `text_encoders/` |
| `t5xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` |

**2. 一次性导出 Kontext 工作流** → `comfyui/flux_kontext_api.json`
（ComfyUI 里 Browse Templates 找 Flux Kontext，开 Dev mode → Export(API)，像 I2V 那样）。
`frames.py` 按节点 class_type 自动识别注入点（LoadImage/CLIPTextEncode/SaveImage），节点 id 不用对。

**3. 用法**：
- 角色参考图放 `projects/<name>/frames/_ref.png`（对应 project.yaml 的 `character_ref`）。
- 每镜**英文**出图提示词写在 project.yaml 的 `frame_prompt`（Kontext 只懂英文）。
- 跑：`python run.py frames projects/lasttram` → 一次生成所有分镜帧。

> ⚠️ Kontext 提示词只支持英文；运动提示词(prompt)与台词(dialogue)仍用中文。

---

## 架构：哪台机器做什么

| 机器 | 角色 | 要不要装 ComfyUI |
|---|---|---|
| **MacBook（你现在）** | 创作机：写 `project.yaml`、出起始帧、`git push` | ❌ 不装（没 N 卡，Wan 14B 跑不动） |
| **Colab A100 80GB** | 渲染机：跑 ComfyUI 服务 + shotforge 编排 | ✅ 全部在这 |

**所以 Mac 这边没有额外 setup**，照常 `git push`。下面的步骤**全部在 Colab 上执行**。

---

## Phase 1：在 Colab 上跑通 ComfyUI + Wan I2V（手动验证质量）

### 0.（可选）腾空间
我们不再用 diffusers 的 Wan 权重了，可释放约 180GB：
```bash
rm -rf ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-*
```

### 1. 安装 ComfyUI + 下载 Wan 2.2 I2V 模型（~60GB）
```bash
cd /content/shotforge && git pull        # 取到本脚本
python scripts/colab_setup.py
```
脚本会：clone ComfyUI、装依赖、把 4 个模型文件下到正确目录、下载官方工作流到 `/content/wan2_2_i2v_workflow.json`。

下载的文件与位置（**官方 I2V 模板用的是 fp8 模型 + 4 步 Lightning LoRA**，不是 fp16）：

| 文件 | 目录 |
|---|---|
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | `ComfyUI/models/diffusion_models/` |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `ComfyUI/models/diffusion_models/` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `ComfyUI/models/text_encoders/` |
| `wan_2.1_vae.safetensors` | `ComfyUI/models/vae/` |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | `ComfyUI/models/loras/` |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | `ComfyUI/models/loras/` |

> Wan 2.2 是 MoE：**高噪 + 低噪两个 14B 专家**，ComfyUI 会在采样中自动切换。A14B 不需要单独的 CLIP vision 模型（和 2.1 不同）。官方模板配 **lightx2v 4 步 Lightning LoRA**——4 步出片、又快又好。**新加模型后必须重启 ComfyUI**（或界面里刷新）才能扫描到，否则报 "required models are missing / Value not in list"。

### 2. 启动 ComfyUI 服务
在一个**后台 cell / 单独终端**里（要一直开着）：
```bash
python /content/ComfyUI/main.py --listen 0.0.0.0 --port 8188
```

### 3. 从你的 Mac 浏览器访问（cloudflared 隧道）
Colab 端口默认不对外，用 cloudflared 开个临时公网地址：
```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
cloudflared tunnel --url http://localhost:8188
```
它会打印一个 `https://xxxx.trycloudflare.com` 地址——在 Mac 浏览器打开就是 ComfyUI 界面。

### 4. 加载工作流，跑通 s1
1. 界面里把 `/content/wan2_2_i2v_workflow.json` 拖进画布（或菜单 Open）——节点会自动连好。
2. 在 **Load Image** 节点上传你的起始帧（如 `projects/example/frames/s1.jpeg`）。
3. 在正向提示词节点填运动提示词（中文即可，Wan 用 umT5）：
   `夏日晴空下的学校天台，少女靠着栏杆，微风轻拂头发与裙摆，白云缓缓飘动`
4. 先用工作流**默认的** steps / cfg / shift / length 跑一遍（**Queue Prompt**），看质量。
5. 如果动作太猛/崩，降 length（帧数）、把提示词写得更"静"；要更快可后续挂 Lightning 加速 LoRA。

**目标**：确认 ComfyUI 这条路能稳定出不崩坏的高质量视频。确认了再做 Phase 2。

---

## Phase 2：shotforge 通过 API 调 ComfyUI（已实现）

`shotforge/comfy.py` + `generate.py --engine comfy` 已接好——保留 `project.yaml` /
批量 / `tools.stitch`，渲染走 ComfyUI。

**用法**（在 Colab 上，ComfyUI 服务要在跑）：
```bash
# 一个 cell 保持运行 ComfyUI 服务：
python /content/ComfyUI/main.py --listen 0.0.0.0 --port 8188

# shotforge 调它渲染（同机，用 localhost，不需要隧道）：
python -m shotforge.generate --project projects/example --engine comfy --shot s1   # 单镜
python -m shotforge.generate --project projects/example --engine comfy             # 全部
python -m tools.stitch       --project projects/example                            # 拼接
```

对每个镜头：上传起始帧 → 把 prompt/negative/宽高/时长/fps/seed 注入
`comfyui/wan_i2v_api.json` → `POST /prompt` → 轮询 `/history` → 下载 mp4 到 `out/`。

**注意**：
- 工作流文件 `comfyui/wan_i2v_api.json`（ComfyUI 导出的 API 格式）。注入用的节点 id 写在
  `shotforge/comfy.py` 顶部的 `NODE_*`。**重新导出过改动的工作流后**节点 id 可能变，跑
  `python -m shotforge.comfy comfyui/wan_i2v_api.json` 打印所有节点对照更新。
- 默认 `--comfy-url http://127.0.0.1:8188`、`--workflow comfyui/wan_i2v_api.json`。
- 当前工作流是 **20 步高质量**路径（`Enable 4steps LoRA?=False`）。想 4 步快出，在 ComfyUI
  里打开那个开关、重新导出 API JSON 覆盖即可。
- 尺寸按起始帧长宽比自动推导（用 `project.yaml` 里 model 的面积预算），注入 WanImageToVideo。

---

## 排错 / 注意

- **磁盘**：ComfyUI 模型 ~60GB，加上原 diffusers 缓存可能吃紧——先做第 0 步腾空间。
- **下载慢/限速**：设 `export HF_TOKEN=...` 再跑脚本；或用 `aria2c -x16` 替代 wget 提速。
- **想更省显存/更快**：把脚本里两个 diffusion model 的 `fp16` 换成 `fp8_scaled` 文件（同目录有），A100 支持 fp8。
- **服务要一直开着**：`main.py` 那个进程关了 API 就没了；Colab 会话断开需重跑。
- **不需要 ComfyUI-Manager / 自定义节点**：近期 ComfyUI 原生支持 Wan 2.2，官方工作流开箱即用。
