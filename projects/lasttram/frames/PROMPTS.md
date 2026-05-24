# 《末班电车》起始帧 — 一键生成（Flux Kontext）

不用再一张张复制提示词了。**你只需要提供 1 张角色参考图**，`frames` 阶段用 Flux
Kontext 让 6 个分镜都"同一个人换场景"，自动保持一致。

每镜的出图提示词已经写进 `project.yaml` 的 **`frame_prompt`（英文，Kontext 只懂英文）**，
角色设定在 `project.yaml` 的 `character`。所以这份文档只剩一件事：**怎么准备那张参考图**。

## 1) 准备 1 张角色参考图 → 存为 `frames/_ref.png`

随便用什么工具做都行（即梦 / nano banana / ComfyUI 文生图 / 甚至一张手绘），
**只要是这个角色的清晰半身/全身像**，9:16 或方图都可。角色设定（出图时用）：

> 日系动漫赛璐璐风格；约 18 岁少女，黑色齐耳短发（波波头），米色风衣 + 浅色围巾，
> 清澈的眼睛；干净通透。

> 这张图决定了全片人物长相——挑一张你最满意的。后面 6 镜都会"长得像它"。

## 2) 一键生成全部 6 张分镜帧

ComfyUI 起着 + 装好 Kontext 模型后（见 `docs/COMFYUI.md` / `scripts/flux_setup.py`）：
```bash
python run.py frames projects/lasttram          # 生成 s1.jpeg ~ s6.jpeg
python run.py frames projects/lasttram --shot s1   # 只重做某一张
```
生成后逐张看一眼（质量门），不满意就改 `project.yaml` 里那镜的 `frame_prompt` 重做。

## 3) 满意后进入视频

```bash
python run.py video projects/lasttram --shot s1   # 先单镜
python run.py video projects/lasttram             # 全部
python run.py dub projects/lasttram && python run.py subs projects/lasttram
python run.py post projects/lasttram --crossfade 0.5 --fade 0.6
```

> 提示：6 镜的取景/朝向已在 `frame_prompt` 里区分（拉远/侧脸/横移/手机特写/抬头微笑/拉远），
> 一致性靠参考图 + Kontext 保证。
