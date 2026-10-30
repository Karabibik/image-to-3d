"""Turnaround sheets: split a sheet into views and frame them the way multi-view models expect."""

from __future__ import annotations

import numpy as np
from PIL import Image

# View slots, named by what the picture shows. Hunyuan3D-2mv's view keys use the same names:
# its upstream example "left" image shows the figure's left side, facing image-left.
SLOTS = ("front", "left", "back", "right")
SLOT_LABELS = {"front": "Front", "left": "Left side", "back": "Back", "right": "Right side"}

ALPHA_THRESHOLD = 16
# Share of the canvas the figure's longest side spans.
FRAME_FILL = 0.9


def _runs(occupied: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """[start, end) runs of True, merging runs separated by fewer than `min_gap` False cells."""
    runs: list[list[int]] = []
    for index in np.flatnonzero(occupied):
        if runs and index - runs[-1][1] < min_gap:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1])
    return [(start, end) for start, end in runs]


def _mask(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA").getchannel("A")) > ALPHA_THRESHOLD


def crop_to_alpha(image: Image.Image) -> Image.Image:
    """Crop an RGBA image to its visible pixels."""
    rows, cols = np.nonzero(_mask(image))
    if not len(rows):
        raise ValueError("the view is empty (fully transparent)")
    return image.convert("RGBA").crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))


def split_sheet(image: Image.Image, min_height_ratio: float = 0.3) -> list[Image.Image]:
    """Cut an RGBA sheet with figures side by side into one crop per figure, left to right.

    Figures are column runs of visible pixels. Inside each run only the tallest row run is
    kept, which drops captions above or below a figure; runs much shorter than the tallest
    figure (stray captions between figures) are dropped too.
    """
    rgba = image.convert("RGBA")
    mask = _mask(rgba)
    width = mask.shape[1]
    figures = []
    for left, right in _runs(mask.any(axis=0), min_gap=max(2, width // 100)):
        rows = _runs(mask[:, left:right].any(axis=1), min_gap=max(2, mask.shape[0] // 100))
        top, bottom = max(rows, key=lambda run: run[1] - run[0])
        figures.append((left, top, right, bottom))
    if not figures:
        raise ValueError("no figures found; is the background transparent or removable?")
    tallest = max(bottom - top for _, top, _, bottom in figures)
    crops = []
    for left, top, right, bottom in figures:
        if bottom - top >= min_height_ratio * tallest:
            crops.append(crop_to_alpha(rgba.crop((left, top, right, bottom))))
    return crops


def align_views(views: dict[str, Image.Image], size: int = 1024) -> dict[str, Image.Image]:
    """Frame every view the same way: equal figure height, centred, on a square canvas.

    Turnaround views show one figure, so they share its height; normalising heights first
    lets views drawn or uploaded at different sizes agree. The longest extent across all
    views then fills FRAME_FILL of the canvas, centred like an object at the origin.
    """
    crops = {slot: crop_to_alpha(image) for slot, image in views.items()}
    height = max(crop.height for crop in crops.values())
    crops = {
        slot: crop.resize((max(1, round(crop.width * height / crop.height)), height), Image.Resampling.LANCZOS)
        for slot, crop in crops.items()
    }
    scale = size * FRAME_FILL / max(height, *(crop.width for crop in crops.values()))
    framed = {}
    for slot, crop in crops.items():
        w, h = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(crop.resize((w, h), Image.Resampling.LANCZOS), ((size - w) // 2, (size - h) // 2))
        framed[slot] = canvas
    return framed
