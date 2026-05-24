# 角色库（可复用的"演员"）

把**角色和剧本解耦**：角色是可复用资产，剧本只管"选角"。换演员 = 改一行。

## 结构
```
characters/<id>/
  character.yaml     # name / appearance(英文) / personality / tone / voice / ref / lora
  ref.png            # 参考图（genref 生成，或自己放）
  lora.safetensors   # 可选，训了角色 LoRA 再放（Phase 2）
```
`character.yaml` 字段：
- **appearance**（英文）：给 SDXL 出参考图 + Flux Kontext 出分镜；会拼到每镜 `frame_prompt` 前。
- **personality / tone**：写台词、选角参考；tone 影响台词语气。
- **voice**：edge-tts 声音 id（配音用）。
- **ref / lora**：参考图、可选 LoRA 的路径（相对角色目录）。

## 选角
在 `projects/<drama>/project.yaml` 写一行：
```yaml
cast: yuki
```
→ 该角色的 **appearance / 参考图 / 声音** 自动带入：
- `frames` 用她的 appearance + ref（Flux Kontext 保持跨镜一致）
- `dub` 用她的 voice
- 写台词时参考她的 personality / tone

（也可在 project.yaml 显式写 `character:` / `character_ref:` 覆盖 cast 的值。）

## 一键生成参考图（开源）
```bash
python run.py genref yuki     # SDXL 动漫从 appearance 出 characters/yuki/ref.png
```
需装 SDXL 动漫模型（`scripts/flux_setup.py` 含 Animagine XL）+ 导出 SDXL txt2img 的 API 工作流到
`comfyui/sdxl_txt2img_api.json`（见 docs/COMFYUI.md）。

## 选角建议（可让我做）
给我剧本，我按各角色的 personality / appearance **推荐谁出演**最合适。

## Phase 2：角色 LoRA（铁一致，跨剧情复用）
1 张参考图 → 用 Kontext 生成 ~15–20 张不同角度/表情的一致图 → 训一个 LoRA
（kohya/sd-scripts，A100 几十分钟）→ 把路径填进 `character.yaml` 的 `lora` → 任意剧情/姿势都铁一致。
高频复用的主角值得做；普通角色用「参考图 + Kontext」已够。

## 完整一轮（用角色库）
```bash
python run.py genref yuki                         # 出/更新角色参考图（一次）
python run.py frames projects/lasttram            # 用 cast 的 yuki 出所有分镜帧
python run.py video  projects/lasttram            # 渲染
python run.py dub    projects/lasttram            # 用 yuki 的声音配音
python run.py subs   projects/lasttram
python run.py post   projects/lasttram --crossfade 0.5 --fade 0.6
```
