"""Character library — reusable "actors" you cast into different stories.

Each character lives in ``characters/<id>/character.yaml`` (a sibling of
``projects/``) with appearance / personality / tone / voice + a reference image
(and optionally a trained LoRA). A project casts one with ``cast: <id>`` in its
project.yaml; the renderer then pulls that character's appearance + reference for
frame generation and voice for dubbing — change the actor by changing one line.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


@dataclass
class Character:
    id: str
    name: str
    appearance: str           # English visual description (for T2I / Flux Kontext)
    personality: str = ""     # for writing dialogue / casting suggestions
    tone: str = ""            # speaking style (influences dialogue voice)
    voice: str = ""           # edge-tts voice id (for tools.dub)
    ref: str = ""             # absolute path to the reference image (may not exist yet)
    lora: str = ""            # absolute path to a trained LoRA (optional, Phase 2)
    lora_trigger: str = ""    # token that activates the LoRA in prompts (defaults to the id)


def characters_dir() -> str:
    """The repo-level ``characters/`` directory (sibling of ``projects/``)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "characters")


def load_character(char: str) -> Character:
    """Load a character by id (``characters/<id>/``) or by an explicit dir path."""
    cdir = char if os.path.isdir(char) else os.path.join(characters_dir(), char)
    manifest = os.path.join(cdir, "character.yaml")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"character not found: {manifest}")
    with open(manifest, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    ref, lora = data.get("ref"), data.get("lora")
    return Character(
        id=os.path.basename(os.path.normpath(cdir)),
        name=str(data.get("name", "")),
        appearance=str(data.get("appearance", "")),
        personality=str(data.get("personality", "")),
        tone=str(data.get("tone", "")),
        voice=str(data.get("voice", "")),
        ref=os.path.join(cdir, str(ref)) if ref else "",
        lora=os.path.join(cdir, str(lora)) if lora else "",
        lora_trigger=str(data.get("lora_trigger", "") or os.path.basename(os.path.normpath(cdir))),
    )


def list_characters() -> list[str]:
    d = characters_dir()
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, "character.yaml")))
