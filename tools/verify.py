"""Auto-verify generated frames against their intent with a vision model (Gemini) —
the "does it match / is it good enough?" gate in the local review loop. Runs on the
Mac (API), flags shots to regenerate and suggests a prompt tweak. Closes the loop:
frames (remote GPU) -> verify (here) -> fix prompts -> regenerate.

    python -m tools.verify --project projects/freefall
    python run.py verify projects/freefall

Needs GEMINI_API_KEY (same as nano banana) + `pip install google-genai pillow`.
$VERIFY_MODEL overrides the model (default gemini-2.5-flash).
"""
from __future__ import annotations

import argparse
import os
import re

DEFAULT_MODEL = os.environ.get("VERIFY_MODEL", "gemini-2.5-flash")

RUBRIC = (
    "你是 AI 出图质检员。下面这张图本应表现这个镜头意图：\n「{intent}」\n\n"
    "请评估：① 画面是否符合该意图（场景/动作/构图/角色）；② 有无明显瑕疵"
    "（畸形的手或脸、扭曲的肢体、多余的人或肢体、文字或水印、糊/低清）。\n"
    "严格只输出三行，不要别的：\n"
    "SCORE: <0到5的整数，5=完美可直接用，<4=建议重出>\n"
    "问题: <一句话；没有就写 无>\n"
    "建议: <一句话，怎么改提示词会更好；没问题就写 无>"
)
_SCORE = re.compile(r"SCORE:\s*([0-5])")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify generated frames against intent (Gemini vision).")
    ap.add_argument("--project", required=True)
    ap.add_argument("--shot", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=int, default=4, help="score below this -> flag for regen")
    a = ap.parse_args()

    from PIL import Image
    try:
        from google import genai
    except ImportError:
        raise SystemExit("[verify] 缺 SDK： pip install google-genai pillow")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("[verify] 需要 GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    from shotforge.manifest import load_project
    project = load_project(a.project)
    print(f"[verify] {project.name} | model={a.model} | 阈值<{a.threshold} 判不合格\n")

    fails = []
    for s in project.shots:
        if a.shot and s.id != a.shot:
            continue
        if not os.path.isfile(s.frame):
            print(f"[skip] {s.id}: 没有帧 {s.frame}")
            continue
        ctx = [f"角色{r}：{project.cast_elements[r].prompt}" for r in s.subjects if project.cast_elements.get(r)]
        intent = s.frame_prompt + ("（" + "；".join(ctx) + "）" if ctx else "") + (f"；画风：{project.style}" if project.style else "")
        try:
            resp = client.models.generate_content(
                model=a.model, contents=[Image.open(s.frame).convert("RGB"), RUBRIC.format(intent=intent)])
            text = (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {s.id}: 调用失败 {str(exc)[:120]}")
            continue
        m = _SCORE.search(text)
        score = int(m.group(1)) if m else -1
        mark = "✓" if score >= a.threshold else "✗"
        print(f"{mark} {s.id}  {text}\n")
        if score < a.threshold:
            fails.append((s.id, score))

    if fails:
        ids = " ".join(f"{i}({sc})" for i, sc in fails)
        print(f"[verify] 建议重出（分数）：{ids}")
        print(f"  例如： python run.py frames {a.project} --shot {fails[0][0]} --overwrite   # 改完提示词后")
    else:
        print("[verify] 全部通过 ✓ —— 可以进入 video 阶段")


if __name__ == "__main__":
    main()
