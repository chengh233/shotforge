"""Unified asset-library loader.

Every library entry — character / scene / style / (later) weather, lighting … —
loads through here into ONE ``Element``: a named bundle of OPTIONAL *contributions*
(a prompt fragment, a negative, a reference image, a LoRA, a motion fragment, a
voice). What an element fills depends on its kind; the *composer* decides how each
contribution is folded into a generation spec. This is the "unify the interface,
not the treatment" design: storage + shape are uniform, composition is role-aware.

Adding a new element TYPE = add a ``<kind>s/<id>/<kind>.yaml`` folder; map its
fields here. No framework code embeds the actual prompts / images / loras.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Element:
    id: str
    kind: str                 # "character" | "scene" | "style" | …
    name: str = ""
    prompt: str = ""          # positive image-prompt fragment (appearance / description / positive)
    negative: str = ""        # negative image-prompt fragment
    ref: str = ""             # absolute path to a reference image (may not exist yet)
    lora: str = ""            # absolute path to a LoRA file (may not exist yet)
    lora_trigger: str = ""    # token that activates the LoRA
    motion: str = ""          # Chinese fragment appended to the Wan motion prompt
    voice: str = ""           # TTS voice id / reference-audio path
    raw: dict = field(default_factory=dict)   # the full yaml (type-specific extras)


# kind -> (yaml filename, which yaml field becomes Element.prompt)
_SPEC = {
    "character": ("character.yaml", "appearance"),
    "scene":     ("scene.yaml",     "description"),
    "style":     ("style.yaml",     "positive"),
    "weather":   ("weather.yaml",   "prompt"),
    "lighting":  ("lighting.yaml",  "prompt"),
}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lib_dir(kind: str) -> str:
    """The repo-level library dir for a kind (``characters/``, ``scenes/`` …)."""
    return os.path.join(_repo_root(), f"{kind}s")


def load_element(kind: str, ident: str) -> Element:
    """Load ``<kind>s/<ident>/<kind>.yaml`` (or an explicit dir) into an Element."""
    import yaml

    fname, prompt_field = _SPEC.get(kind, (f"{kind}.yaml", "prompt"))
    edir = ident if os.path.isdir(ident) else os.path.join(lib_dir(kind), ident)
    manifest = os.path.join(edir, fname)
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"{kind} not found: {manifest}")
    with open(manifest, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    def _rel(v):  # join a relative asset path to the element dir
        return os.path.join(edir, str(v)) if v else ""

    return Element(
        id=os.path.basename(os.path.normpath(edir)),
        kind=kind,
        name=str(data.get("name", "")),
        prompt=str(data.get(prompt_field, "") or ""),
        negative=str(data.get("negative", "") or ""),
        ref=_rel(data.get("ref")),
        lora=_rel(data.get("lora")) if data.get("lora") not in (None, "null") else "",
        lora_trigger=str(data.get("lora_trigger", "") or ""),
        motion=str(data.get("video_suffix", data.get("motion", "")) or ""),
        voice=str(data.get("voice", "") or ""),
        raw=data,
    )


def list_elements(kind: str) -> list[str]:
    d = lib_dir(kind)
    fname = _SPEC.get(kind, (f"{kind}.yaml",))[0]
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, fname)))
