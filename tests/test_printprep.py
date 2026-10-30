import numpy as np
import pytest
import trimesh

from app.printprep import PrintSettings, load_geometry, nonmanifold_edge_count, prepare, solidify


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
    # A duplicated face makes three edges non-manifold, as in messy generated meshes. The old
    # repair path deleted faces and flipped patches inside-out; now it must be remeshed.
    box = trimesh.creation.box(extents=[1.0, 0.6, 2.0])
    faces = np.vstack([box.faces, box.faces[:1]])
    broken = trimesh.Trimesh(box.vertices, faces, process=False)
    src = tmp_path / "broken.glb"
    broken.export(src)

    report = prepare(src, tmp_path / "out", PrintSettings(
        source_up="z", target_height_mm=20, remesh_cell_mm=0.2, flatten_base_mm=0.3, target_faces=None,
    ))
    assert report.remeshed and report.flattened
    assert report.watertight and report.winding_consistent and report.nonmanifold_edges == 0
    mesh = trimesh.load(report.files["stl"], force="mesh")
    assert mesh.volume > 0  # outward-facing
    assert mesh.extents[2] == pytest.approx(20.0, rel=0.02)


def _scrambled_shell() -> trimesh.Trimesh:
    """An unoriented open shell: holes, faces wound at random."""
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=10.0)
    # Two holes 44% of the diameter wide. Holes of 60% still give a closed solid, with deep dents.
    keep = (sphere.triangles_center[:, 2] < 9.0) & (sphere.triangles_center[:, 0] < 9.0)
    faces = sphere.faces[keep].copy()
    flip = np.random.default_rng(0).random(len(faces)) < 0.5
    faces[flip] = faces[flip][:, ::-1]
    return trimesh.Trimesh(sphere.vertices, faces, process=False)


def test_solidify_fills_an_open_unoriented_shell():
    shell = _scrambled_shell()
    assert not shell.is_watertight and not shell.is_winding_consistent
    solid = solidify(shell, 0.25)
    assert solid.is_watertight and solid.is_winding_consistent and solid.volume > 0
    assert nonmanifold_edge_count(solid) == 0
    assert solid.volume == pytest.approx(4 / 3 * np.pi * 10.0 ** 3, rel=0.12)  # a ball, not a shell


def test_solidify_keeps_concavities_gaps_and_thin_parts():
    # A wide shallow bowl, broken by deleting one face.
    cube = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    pit = trimesh.creation.box(extents=[16.0, 16.0, 10.0])
    pit.apply_translation([0, 0, 8.0])
    bowl = trimesh.boolean.difference([cube, pit], engine="manifold").subdivide().subdivide().subdivide()
    bowl.update_faces(np.arange(len(bowl.faces)) != 0)
    assert solidify(bowl, 0.1).volume == pytest.approx(cube.volume - 16.0 ** 2 * 7, rel=0.08)

    # A narrow slot right through a block stays open: light passes both ways along it.
    block = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    slot = trimesh.creation.box(extents=[2.0, 30.0, 8.0])
    slotted = trimesh.boolean.difference([block, slot], engine="manifold").subdivide().subdivide().subdivide()
    slotted.update_faces(np.arange(len(slotted.faces)) != 0)
    assert solidify(slotted, 0.1).volume == pytest.approx(block.volume - 2.0 * 20.0 * 8.0, rel=0.08)

    # A single open sheet stays as a thin plate instead of vanishing.
    plate = trimesh.creation.box(extents=[20.0, 20.0, 1e-3]).subdivide().subdivide()
    plate.update_faces(plate.triangles_center[:, 2] > 0)
    assert not plate.is_watertight
    sheet = solidify(plate, 0.25)
    assert sheet.is_watertight
    assert sheet.extents[:2] == pytest.approx([20.0, 20.0], abs=0.6) and sheet.extents[2] < 1.5


def test_open_shell_prints_solid(tmp_path):
    shell = _scrambled_shell()
    src = tmp_path / "shell.glb"
    shell.export(src)
    report = prepare(src, tmp_path / "out", PrintSettings(  # holes too big for the ordinary repair
        source_up="z", target_height_mm=30, remesh_cell_mm=0.25, flatten_base_mm=0.5, target_faces=20000,
        close_holes_max_edges=3,
    ))
    assert report.remeshed and report.watertight and report.flattened and report.nonmanifold_edges == 0
    assert not any("hollow" in w for w in report.warnings)
    assert report.size_mm[2] == pytest.approx(30, rel=0.03)


def test_fragmented_surface_keeps_small_real_parts(tmp_path):
    # An open surface can be many fragments, so a small part next to the body must survive
    # until solidify() has joined them; only then is a floater really a floater.
    body = _scrambled_shell()
    handle = trimesh.creation.box(extents=[1.5, 1.5, 6.0]).subdivide().subdivide()
    handle.apply_translation([0, 0, 12.5])  # touches the top of the body
    handle.update_faces(np.arange(len(handle.faces)) != 0)
    floater = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    floater.apply_translation([14.0, 14.0, 14.0])
    src = tmp_path / "fragments.glb"
    trimesh.util.concatenate([body, handle, floater]).export(src)
    report = prepare(src, tmp_path / "out", PrintSettings(
        source_up="z", target_height_mm=None, remesh_cell_mm=0.25, flatten_base_mm=0, target_faces=None,
        close_holes_max_edges=3,
    ))
    assert report.remeshed and report.watertight
    assert report.parts_removed == 1  # the floater, and only it
    assert report.size_mm[2] == pytest.approx(25.5, abs=1.0)  # the handle is still on top


def test_hollow_shell_is_flagged(tmp_path, figure_glb):
    # One connected, watertight thin-walled box (a tunnel joins the cavity to the outside),
    # like the double-skinned meshes a narrow-band remesh produces.
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
