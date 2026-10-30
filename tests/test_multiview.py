import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.multiview import FRAME_FILL, align_views, split_sheet


def _bbox(image: Image.Image) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(np.asarray(image.getchannel("A")) > 16)
    return cols.min(), rows.min(), cols.max() + 1, rows.max() + 1


def test_split_sheet_finds_figures_left_to_right_and_drops_captions():
    sheet = Image.new("RGBA", (900, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    colours = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
    for i, colour in enumerate(colours):
        x = 60 + i * 300
        draw.rectangle((x, 40, x + 90 + 30 * i, 320), fill=colour)  # figures of different widths
        draw.rectangle((x, 350, x + 60, 370), fill=(10, 10, 10, 255))  # caption under each figure
    draw.rectangle((330, 180, 345, 190), fill=(10, 10, 10, 255))  # stray mark between figures

    crops = split_sheet(sheet)
    assert len(crops) == 3
    for crop, colour, extra in zip(crops, colours, (0, 30, 60)):
        assert crop.size == (91 + extra, 281)  # caption gone, cropped tight
        assert crop.getpixel((crop.width // 2, crop.height // 2)) == colour


def test_split_sheet_needs_visible_figures():
    with pytest.raises(ValueError):
        split_sheet(Image.new("RGBA", (100, 100), (0, 0, 0, 0)))


def test_align_views_share_height_and_centre():
    front = Image.new("RGBA", (500, 800), (0, 0, 0, 0))
    ImageDraw.Draw(front).rectangle((150, 100, 350, 700), fill=(200, 50, 50, 255))
    side = Image.new("RGBA", (120, 160), (0, 0, 0, 0))  # same figure, uploaded smaller
    ImageDraw.Draw(side).rectangle((40, 20, 80, 140), fill=(50, 200, 50, 255))

    framed = align_views({"front": front, "left": side}, size=512)
    boxes = {slot: _bbox(image) for slot, image in framed.items()}
    for slot, (x0, y0, x1, y1) in boxes.items():
        assert framed[slot].size == (512, 512)
        assert y1 - y0 == pytest.approx(512 * FRAME_FILL, abs=2)  # height is the longest extent
        assert (x0 + x1) / 2 == pytest.approx(256, abs=1.5) and (y0 + y1) / 2 == pytest.approx(256, abs=1.5)
    front_w, side_w = boxes["front"][2] - boxes["front"][0], boxes["left"][2] - boxes["left"][0]
    assert front_w / side_w == pytest.approx(200 / 600 / (40 / 120), rel=0.05)


def test_wide_figures_fit_the_frame():
    wide = Image.new("RGBA", (400, 100), (0, 0, 0, 0))
    ImageDraw.Draw(wide).rectangle((0, 0, 399, 99), fill=(0, 0, 0, 255))
    x0, y0, x1, y1 = _bbox(align_views({"front": wide}, size=256)["front"])
    assert x1 - x0 == pytest.approx(256 * FRAME_FILL, abs=2)
