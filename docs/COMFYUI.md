# ComfyUI 渲染后端（方案 B）

diffusers 的 Wan 图生视频出图融化/崩坏（我们试了 fp32 VAE、CLIPVisionModel、auto-resolution、升级 diffusers 都没解决——这是 diffusers Wan I2V 实现本身的已知质量问题）。**ComfyUI 用的是另一套、社区公认更好的 Wan 实现**，是高质量 Wan 视频的主流路径。

方案 B = **shotforge 继续做编排（剧本/批量/拼接），渲染换成调 ComfyUI**。分两步：先在 Colab 上把 ComfyUI 跑通、手动验证质量（Phase 1），再把 shotforge 接到它的 API（Phase 2）。

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
bash scripts/comfyui_setup.sh
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

## Phase 2：shotforge 通过 API 调 ComfyUI（待 Phase 1 验证后做）

思路：保留 `project.yaml` / 批量 / `tools.stitch`，把渲染后端从 diffusers 换成「调 ComfyUI 的 HTTP API」：

1. 把 Phase 1 验证好的工作流导出为 **API 格式 JSON**（ComfyUI 设置里开 "Enable Dev mode Options" → Save (API Format)）。存进 `projects/` 或 `shotforge/`。
2. 新增一个 `shotforge/comfy.py`：读该 workflow JSON，按每个镜头**注入** start image / prompt / 帧数 / 尺寸 / seed，POST 到 `http://127.0.0.1:8188/prompt`，轮询 `/history/{id}`，取回输出帧/mp4。
3. `generate.py` 增加一个 `--backend comfy`（或在 `project.yaml` 加 `engine: comfy`），走 ComfyUI 而非 diffusers。
4. ComfyUI 和 shotforge 在**同一台 Colab**上，所以用 `localhost:8188`，不需要隧道（隧道只是给你的 Mac 浏览器看 GUI 用的）。

> Phase 2 是一段代码活，依赖 Phase 1 跑通后导出的**确切 API workflow JSON**（节点 id 要对上才能注入参数）。**先把 Phase 1 跑通、把那个 API JSON 发我，我来写 `shotforge/comfy.py` 和接线。**

---

## 排错 / 注意

- **磁盘**：ComfyUI 模型 ~60GB，加上原 diffusers 缓存可能吃紧——先做第 0 步腾空间。
- **下载慢/限速**：设 `export HF_TOKEN=...` 再跑脚本；或用 `aria2c -x16` 替代 wget 提速。
- **想更省显存/更快**：把脚本里两个 diffusion model 的 `fp16` 换成 `fp8_scaled` 文件（同目录有），A100 支持 fp8。
- **服务要一直开着**：`main.py` 那个进程关了 API 就没了；Colab 会话断开需重跑。
- **不需要 ComfyUI-Manager / 自定义节点**：近期 ComfyUI 原生支持 Wan 2.2，官方工作流开箱即用。
