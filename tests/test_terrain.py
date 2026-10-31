from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from app import terrain
from app.terrain import TileSettings, camera_height, cut_tile, export_world, height_grid, load_ground, solid_tile

CAMERA_HEIGHT = 1.7


def ground_layer(size: float = 40.0, cells: int = 240) -> trimesh.Trimesh:
    """Flat ground below the camera, with a hill centred at (10, 0) and the camera at the origin."""
    xs = np.linspace(-size / 2, size / 2, cells)
    x, y = np.meshgrid(xs, xs)
    z = -CAMERA_HEIGHT + 2.0 * np.exp(-((x - 10) ** 2 + y ** 2) / 8)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    index = np.arange(cells * cells).reshape(cells, cells)
    a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    colors = np.tile([90, 140, 70, 255], (len(vertices), 1))
    return trimesh.Trimesh(vertices, faces, vertex_colors=colors, process=False)


def house_layer() -> trimesh.Trimesh:
    """A 2 x 2 box standing 3 high on the ground, centred at (-8, 0)."""
    box = trimesh.creation.box(extents=(2, 2, 3))
    box.apply_translation([-8, 0, -CAMERA_HEIGHT + 1.5])
    box.visual.vertex_colors = [160, 80, 60, 255]
    return box


def sky_layer() -> trimesh.Trimesh:
    sky = trimesh.creation.icosphere(subdivisions=4, radius=60.0)
    sky.visual.vertex_colors = [120, 170, 230, 255]
    return sky


@pytest.fixture
def world(tmp_path) -> list[dict]:
    layers = []
    for index, (kind, mesh) in enumerate([("fg1", house_layer()), ("bg", ground_layer()), ("sky", sky_layer())]):
        path = tmp_path / f"mesh_layer{index}.ply"
        mesh.export(path)
        layers.append({"type": kind, "path": str(path)})
    return layers


def units(meters: float) -> float:
    return meters / CAMERA_HEIGHT


def test_the_ground_leaves_the_sky_out(world):
    ground = load_ground(world)
    assert ground.bounds[1, 2] < 10  # the sky reaches 60
    assert len(ground.visual.vertex_colors) == len(ground.vertices)


def test_a_world_without_ground_is_refused(world):
    with pytest.raises(ValueError, match="no ground"):
        load_ground([layer for layer in world if layer["type"] == "sky"])


def test_camera_height_is_read_from_the_ground_below(world):
    assert camera_height(np.asarray(load_ground(world).vertices)) == pytest.approx(CAMERA_HEIGHT, abs=0.01)


def test_a_tile_is_a_watertight_solid_of_the_exact_width(world, tmp_path):
    settings = TileSettings(center_x=units(10), size=units(12), grid=200, width_mm=50, base_mm=3)
    report = cut_tile(world, CAMERA_HEIGHT, settings, tmp_path / "tile")
    assert report.watertight
    assert report.nonmanifold_edges == 0
    assert report.size_mm[0] == pytest.approx(50, abs=0.01)
    assert report.size_mm[1] == pytest.approx(50, abs=0.01)
    reloaded = trimesh.load(report.files["stl"])
    reloaded.merge_vertices()
    assert reloaded.is_watertight
    assert reloaded.is_volume


def test_a_tile_keeps_the_real_height_of_what_stands_on_the_ground(world, tmp_path):
    settings = TileSettings(center_x=units(-8), size=units(6), grid=240, width_mm=60, base_mm=2, smoothing=0)
    report = cut_tile(world, CAMERA_HEIGHT, settings, tmp_path / "tile")
    # 6 m on the ground is 60 mm, so the 3 m house is 30 mm above the 2 mm base.
    assert report.size_mm[2] == pytest.approx(32, abs=0.3)


def test_the_sky_is_not_in_the_tile(world, tmp_path):
    settings = TileSettings(center_x=units(-16), center_y=units(14), size=units(5), grid=100, width_mm=50, base_mm=3)
    report = cut_tile(world, CAMERA_HEIGHT, settings, tmp_path / "tile")
    assert report.size_mm[2] == pytest.approx(3, abs=0.2)  # flat ground, just the base


def test_rotating_a_tile_turns_the_ground_under_it(world):
    ground = load_ground(world)
    ground.apply_scale(1 / CAMERA_HEIGHT)
    base = TileSettings(center_x=units(10), size=units(14), grid=64, smoothing=0)
    plain, _ = height_grid(ground, base)
    turned, _ = height_grid(ground, TileSettings(**{**base.__dict__, "rotation_deg": 90}))
    assert np.abs(np.rot90(turned, 1) - plain).max() < 0.05


def test_ground_beyond_the_edge_of_the_world_is_flagged(world, tmp_path):
    settings = TileSettings(center_x=units(20), size=units(8), grid=64)
    report = cut_tile(world, CAMERA_HEIGHT, settings, tmp_path / "tile")
    assert report.watertight
    assert any("real ground" in warning for warning in report.warnings)


def test_a_tile_with_no_ground_is_refused(world, tmp_path):
    with pytest.raises(ValueError, match="no ground"):
        cut_tile(world, CAMERA_HEIGHT, TileSettings(center_x=units(200), size=units(4), grid=32), tmp_path / "tile")


def test_exaggeration_stretches_the_relief_but_not_the_base(world, tmp_path):
    flat = TileSettings(center_x=units(10), size=units(10), grid=100, width_mm=50, base_mm=3, exaggeration=1)
    tall = TileSettings(**{**flat.__dict__, "exaggeration": 2})
    plain = cut_tile(world, CAMERA_HEIGHT, flat, tmp_path / "plain").size_mm[2]
    double = cut_tile(world, CAMERA_HEIGHT, tall, tmp_path / "tall").size_mm[2]
    assert double - 3 == pytest.approx((plain - 3) * 2, rel=0.02)


def test_the_solid_of_any_grid_is_closed_and_faces_outwards():
    heights = np.random.default_rng(1).random((9, 9))
    solid = solid_tile(heights, 20.0, 2.0)
    assert solid.is_volume
    assert solid.volume > 0
    assert solid.bounds[0, 2] == 0


def test_exporting_a_world(world, tmp_path):
    summary = export_world(world, tmp_path)
    assert summary["unit"] == pytest.approx(CAMERA_HEIGHT, abs=0.01)
    assert Path(summary["map"]).exists()
    merged = trimesh.load(summary["world"], force="mesh")
    assert merged.extents.max() > 100  # the sky is in the full model
    preview = trimesh.load(summary["preview"], force="mesh")
    assert preview.extents.max() == pytest.approx(1.0, abs=0.01)  # framed for the viewer, no sky
    assert len(preview.visual.vertex_colors) == len(preview.vertices)


def test_a_heavy_preview_is_reduced_and_keeps_its_colours(tmp_path, monkeypatch):
    monkeypatch.setattr(terrain, "WORLD_PREVIEW_FACES", 2000)
    layer = tmp_path / "ground.ply"
    ground_layer(cells=100).export(layer)
    summary = export_world([{"type": "bg", "path": str(layer)}], tmp_path)
    preview = trimesh.load(summary["preview"], force="mesh")
    assert len(preview.faces) <= 2200
    assert tuple(preview.visual.vertex_colors[0][:3]) == (90, 140, 70)


def test_the_map_marks_the_tile(world, tmp_path):
    summary = export_world(world, tmp_path)
    before = Image.open(summary["map"]).convert("RGB")
    settings = TileSettings(center_x=units(10), size=units(8))
    marked = terrain.map_with_tile(before, summary["map_half"], settings)
    assert marked.size == before.size
    assert np.abs(np.asarray(marked, dtype=int) - np.asarray(before, dtype=int)).sum() > 0
