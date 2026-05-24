"""Style library — reusable "looks" (the art director). Decouples the visual
style from the script and the assets: instead of repeating "Kyoto Animation
anime style…" in every frame prompt, a project picks one with ``style: <id>``.

A style lives in ``styles/<id>/style.yaml`` (sibling of characters/ and scenes/):
  - ``positive``     appended to image-generation prompts (the look)
  - ``negative``     negative tags for SDXL image generation
  - ``video_suffix`` Chinese suffix appended to the Wan motion prompts
  - ``ref`` / ``lora``  optional style reference image / style LoRA
Switch the whole drama's look by changing one line.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


@dataclass
class Style:
    id: str
    name: str = ""
    positive: str = ""        # appended to image-gen prompts
    negative: str = ""        # negative tags (SDXL image gen)
    video_suffix: str = ""    # Chinese suffix for Wan motion prompts (e.g. 日系动漫电影感)
    ref: str = ""             # optional style reference image
    lora: str = ""            # optional style LoRA


def styles_dir() -> str:
    """The repo-level ``styles/`` directory (sibling of ``characters/``)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "styles")


def load_style(style: str) -> Style:
    """Load a style by id (``styles/<id>/``) or by an explicit dir path."""
    sdir = style if os.path.isdir(style) else os.path.join(styles_dir(), style)
    manifest = os.path.join(sdir, "style.yaml")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"style not found: {manifest}")
    with open(manifest, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    ref, lora = data.get("ref"), data.get("lora")
    return Style(
        id=os.path.basename(os.path.normpath(sdir)),
        name=str(data.get("name", "")),
        positive=str(data.get("positive", "")),
        negative=str(data.get("negative", "")),
        video_suffix=str(data.get("video_suffix", "")),
        ref=os.path.join(sdir, str(ref)) if ref else "",
        lora=os.path.join(sdir, str(lora)) if lora else "",
    )


def list_styles() -> list[str]:
    d = styles_dir()
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, "style.yaml")))
