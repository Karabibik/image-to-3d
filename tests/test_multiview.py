import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from app.multiview import FRAME_FILL, align_views, camera_to_world, split_sheet

# Pixal3D's assets/mv_images/example/transforms.json (distance 3.1192049980163574).
PIXAL_EXAMPLE = {
    "front": [[1, 0, 0, 0], [0, 0, -1, -3.1192049980163574], [0, 1, 0, 0], [0, 0, 0, 1]],
    "left": [[0, 0, 1, 3.1192049980163574], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    "back": [[-1, 0, 0, 0], [0, 0, 1, 3.1192049980163574], [0, 1, 0, 0], [0, 0, 0, 1]],
    "right": [[0, 0, -1, -3.1192049980163574], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
}


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


@pytest.mark.parametrize("slot", PIXAL_EXAMPLE)
def test_cameras_match_pixal3d_example(slot):
    assert np.allclose(camera_to_world(slot, 3.1192049980163574), PIXAL_EXAMPLE[slot], atol=1e-9)


@pytest.mark.parametrize("slot", PIXAL_EXAMPLE)
def test_elevated_cameras_are_the_level_rig_tilted_about_x(slot):
    # Raising a camera by e around the figure is a rotation of the level rig by -e about X
    # for the front camera; every camera must stay orthonormal and look at the origin.
    c2w = np.array(camera_to_world(slot, 2.0, elevation_deg=25))
    rotation, position = c2w[:3, :3], c2w[:3, 3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9) and np.linalg.det(rotation) > 0
    assert np.allclose(-rotation[:, 2], -position / 2.0, atol=1e-9)  # looks at the origin
    assert position[2] == pytest.approx(2.0 * np.sin(np.radians(25)))
    if slot == "front":
        tilt = np.eye(4)
        tilt[:3, :3] = trimesh.transformations.rotation_matrix(np.radians(-25), [1, 0, 0])[:3, :3]
        assert np.allclose(c2w, tilt @ np.array(camera_to_world("front", 2.0)), atol=1e-9)
