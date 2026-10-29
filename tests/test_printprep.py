import numpy as np
import pytest
import trimesh

from app.printprep import PrintSettings, load_geometry, prepare


def _figure_y_up() -> trimesh.Trimesh:
    """A rough miniature: a disc base with a tall capsule body, standing along +Y."""
    base = trimesh.creation.cylinder(radius=0.5, height=0.1, sections=64)
    body = trimesh.creation.capsule(height=1.2, radius=0.25, count=[48, 48])
    body.apply_translation([0, 0, 0.05 + 0.25 + 0.6])
    figure = trimesh.boolean.union([base, body], engine="manifold")
    # Built Z-up; rotate so the figure stands along +Y like glTF output.
    figure.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
    return figure


@pytest.fixture
def figure_glb(tmp_path):
    path = tmp_path / "figure.glb"
    _figure_y_up().export(path)
    return path


def test_orients_scales_and_places(figure_glb, tmp_path):
    report = prepare(figure_glb, tmp_path / "out", PrintSettings(target_height_mm=32.0, target_faces=None))
    mesh = trimesh.load(report.files["stl"], force="mesh")
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] == pytest.approx(32.0, rel=1e-3)
    assert lo[2] == pytest.approx(0.0, abs=1e-6)
    # Standing up: Z must be the longest extent.
    assert mesh.extents[2] == max(mesh.extents)
    assert report.watertight


def test_removes_floaters_and_closes_holes(tmp_path):
    figure = _figure_y_up()
    floater = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    floater.apply_translation([1.0, 1.0, 1.0])
    holed = trimesh.util.concatenate([figure, floater])
    # Punch a small hole in the body.
    holed.update_faces(np.arange(len(holed.faces)) != 10)
    assert not holed.is_watertight
    src = tmp_path / "holed.glb"
    holed.export(src)

    report = prepare(src, tmp_path / "out", PrintSettings(target_faces=None))
    assert report.parts_removed == 1
    assert report.watertight
    assert report.volume_cm3 and report.volume_cm3 > 0


def test_decimates_to_target(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=6)
    src = tmp_path / "sphere.glb"
    sphere.export(src)
    report = prepare(src, tmp_path / "out", PrintSettings(target_faces=10_000))
    assert report.faces == pytest.approx(10_000, rel=0.05)
    assert report.watertight


def test_flatten_base_gives_flat_bottom(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=5)
    src = tmp_path / "sphere.glb"
    sphere.export(src)
    report = prepare(src, tmp_path / "out", PrintSettings(target_height_mm=20, flatten_base_mm=1.0, target_faces=None))
    mesh = trimesh.load(report.files["stl"], force="mesh")
    assert report.flattened
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert mesh.extents[2] == pytest.approx(20.0, rel=1e-3)
    bottom = mesh.area_faces[mesh.face_normals[:, 2] < -0.999].sum()
    assert bottom > 10  # a disc of radius ~4.36 mm has ~60 mm^2
    assert report.watertight


def test_exports_requested_formats(figure_glb, tmp_path):
    report = prepare(figure_glb, tmp_path / "out", PrintSettings(formats=("stl", "3mf")))
    for key in ("stl", "3mf", "preview"):
        assert key in report.files
        assert len(load_geometry(report.files[key]).faces) > 0
