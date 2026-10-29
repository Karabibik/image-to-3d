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


def test_nonmanifold_input_is_remeshed_not_flipped(tmp_path):
    # A duplicated face makes three edges non-manifold, like TRELLIS.2 output. The old
    # repair path deleted faces and flipped patches inside-out; now it must be remeshed.
    box = trimesh.creation.box(extents=[1.0, 0.6, 2.0])
    faces = np.vstack([box.faces, box.faces[:1]])
    broken = trimesh.Trimesh(box.vertices, faces, process=False)
    src = tmp_path / "broken.glb"
    broken.export(src)

    report = prepare(src, tmp_path / "out", PrintSettings(
        source_up="z", target_height_mm=20, remesh_cell_mm=0.2, flatten_base_mm=0.3, target_faces=None,
    ))
    assert any("non-manifold" in w for w in report.warnings)
    assert report.remeshed and report.flattened
    assert report.watertight and report.winding_consistent and report.nonmanifold_edges == 0
    mesh = trimesh.load(report.files["stl"], force="mesh")
    assert mesh.volume > 0  # outward-facing
    assert mesh.extents[2] == pytest.approx(20.0, rel=0.02)


def test_hollow_shell_is_flagged(tmp_path, figure_glb):
    # One connected, watertight thin-walled box (a tunnel joins the cavity to the outside),
    # like the double-skinned meshes TRELLIS.2's narrow-band remesh produces.
    outer = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    cavity = trimesh.creation.box(extents=[0.99, 0.99, 0.99])
    tunnel = trimesh.creation.cylinder(radius=0.05, height=0.6, sections=24)
    tunnel.apply_translation([0, 0, 0.4])
    shell = trimesh.boolean.difference([outer, trimesh.boolean.union([cavity, tunnel], engine="manifold")], engine="manifold")
    src = tmp_path / "shell.glb"
    shell.export(src)
    report = prepare(src, tmp_path / "shell", PrintSettings(source_up="z", target_faces=None))
    assert report.watertight
    assert any("hollow" in w for w in report.warnings)

    solid = prepare(figure_glb, tmp_path / "solid", PrintSettings(target_faces=None))
    assert not any("hollow" in w for w in solid.warnings)


def test_exports_requested_formats(figure_glb, tmp_path):
    report = prepare(figure_glb, tmp_path / "out", PrintSettings(formats=("stl", "3mf")))
    for key in ("stl", "3mf", "preview"):
        assert key in report.files
        assert len(load_geometry(report.files[key]).faces) > 0
