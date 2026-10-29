"""Input image preparation: keep real transparency, otherwise cut the subject out with rembg."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_sessions: dict = {}


def has_transparency(image: Image.Image) -> bool:
    if image.mode not in ("RGBA", "LA", "P"):
        return False
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    return bool((alpha < 250).any())


def remove_background(image: Image.Image, model: str) -> Image.Image:
    from rembg import new_session, remove

    if model not in _sessions:
        _sessions[model] = new_session(model)
    return remove(image.convert("RGB"), session=_sessions[model]).convert("RGBA")


def prepare_input(src: str | Path, dst: str | Path, rembg_model: str) -> tuple[Path, bool]:
    """Write an RGBA cut-out of `src` to `dst`. Returns (path, background_was_removed)."""
    image = ImageOps.exif_transpose(Image.open(src))
    removed = not has_transparency(image)
    out = remove_background(image, rembg_model) if removed else image.convert("RGBA")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    return dst, removed
