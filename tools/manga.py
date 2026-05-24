"""Assemble a project's frames into a manga / storyboard page (条漫) — the
"manga first" check: read the story as STILLS, before animating. If the still
panels already tell the story, the video will too. CPU, runs on the Mac.

    python -m tools.manga --project projects/lasttram
    python run.py manga projects/lasttram

Stacks each shot's frame vertically (webtoon style) with its dialogue typeset
beneath as a caption, and a small shot-id tag. Output: <project>/out/manga.png
"""
from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

# CJK-capable fonts to try (macOS first, then Linux/Noto).
_FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _font(size: int):
    for p in _FONT_PATHS:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble frames into a manga/storyboard page (条漫).")
    ap.add_argument("--project", required=True)
    ap.add_argument("--width", type=int, default=760, help="panel width in px")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shotforge.manifest import load_project

    project = load_project(args.project)
    W = args.width
    margin, gutter, border = 26, 30, 3
    cap_font = _font(max(22, W // 24))
    id_font = _font(max(16, W // 32))

    panels = []
    for shot in project.shots:
        if not os.path.isfile(shot.frame):
            print(f"[manga] missing {shot.frame} — skip {shot.id}")
            continue
        im = Image.open(shot.frame).convert("RGB")
        ph = int(im.size[1] * (W / im.size[0]))
        panels.append((shot.id, im.resize((W, ph)), ph, (shot.dialogue or "").strip()))
    if not panels:
        raise SystemExit("[manga] no frames found — generate frames first")

    cap_h = cap_font.size + 20
    page_w = W + 2 * margin
    page_h = margin + sum(ph + (cap_h if cap else 0) + gutter for _, _, ph, cap in panels) - gutter + margin
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)

    y = margin
    for sid, im, ph, cap in panels:
        x = margin
        draw.rectangle([x - border, y - border, x + W + border - 1, y + ph + border - 1],
                       outline="black", width=border)
        page.paste(im, (x, y))
        draw.text((x + 10, y + 8), sid, fill="white", font=id_font, stroke_width=2, stroke_fill="black")
        y += ph
        if cap:
            draw.text((x + 4, y + 10), cap, fill="black", font=cap_font)
            y += cap_h
        y += gutter

    out = args.out or os.path.join(project.root, "out", "manga.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    page.save(out)
    print(f"[manga] {len(panels)} panels -> {out}  ({page_w}x{page_h})")


if __name__ == "__main__":
    main()
