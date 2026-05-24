"""Image engine: Google Nano Banana (Gemini 2.5 Flash Image). Reference-conditioned,
runs on the Mac (API). Reuses the proven helpers in tools/nanobanana.py — this is
just the thin engine adapter (content-free; the prompt/refs come from the composer).
Ignores spec.loras (Nano Banana has no LoRA input)."""
from __future__ import annotations

from .base import ImageSpec


class NanoBananaEngine:
    name = "nanobanana"

    def __init__(self) -> None:
        self._client = None
        self._model = None

    def generate(self, spec: ImageSpec) -> None:
        import os
        from tools.nanobanana import DEFAULT_MODEL, _client, _load_refs, generate
        if self._client is None:
            self._client = _client(None)
        # read at generate-time so project `engines: image_model:` / $NANOBANANA_MODEL apply
        model = os.environ.get("NANOBANANA_MODEL") or DEFAULT_MODEL
        refs = _load_refs(spec.refs)
        generate(self._client, model, spec.prompt, refs, spec.out)


ENGINE = NanoBananaEngine()
