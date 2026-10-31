"""Turn a generated mesh into a print-ready model.

Steps, in order: load geometry only, convert to Z-up, repair and close holes, scale to a
target height in millimetres, rebuild as a solid if the mesh is still open, drop floaters,
optionally flatten the base, decimate to a face budget, export STL/3MF plus a Y-up GLB for
the browser preview.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fast_simplification
import manifold3d
import numpy as np
import pymeshlab
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

log = logging.getLogger(__name__)


@dataclass
class PrintSettings:
    # Up axis of the incoming mesh. Every model in this project writes glTF, which is Y-up.
    source_up: str = "y"
    # Manual corrections applied after the up-axis conversion, in degrees.
    rotate_x_deg: float = 0.0
    rotate_z_deg: float = 0.0
    # Loose parts with fewer faces than this fraction of the largest part are deleted.
    min_component_ratio: float = 0.02
    close_holes_max_edges: int = 2000
    # Output height along Z in millimetres; None keeps the model's own units.
    target_height_mm: float | None = 32.0
    # Cut this many millimetres off the bottom so the base sits flat; 0 disables.
    flatten_base_mm: float = 0.0
    # Solid rebuild (cell in output units) when the mesh is still not watertight after repair,
    # e.g. Hi3D output. Smooths detail finer than the cell.
    remesh_if_open: bool = True
    remesh_cell_mm: float = 0.05
    # Face budget after decimation; None or 0 keeps every face (capped after a remesh).
    target_faces: int | None = 200_000
    formats: tuple[str, ...] = ("stl", "3mf")
    # Fit the preview by its longest side rather than its height, for wide flat models.
    preview_whole: bool = False


# solidify() at 0.05 mm yields 1-2M faces for a 32 mm figure; "keep all" stops here instead.
REMESH_FACE_CAP = 1_000_000
# Step previews: refined meshes are too heavy for a browser.
PREVIEW_FACES = 150_000
# Measured: solid minis fill ~10% of their bounding box, double-walled shells 0.5-1.3%.
HOLLOW_FILL_RATIO = 0.03

# solidify(): the longest side gets at most this many cells, which bounds memory on tall models.
SOLID_MAX_CELLS = 900
# The inside/outside test runs on a grid this many times coarser than the surface grid.
SOLID_COARSE_FACTOR = 4
# Enclosed means at most this many of the 26 grid directions see out. Measured: at 4 and
# above, cells along the walls of a shallow bowl qualify and fill it from the corners; a real
# cavity behind holes stays filled down to 1.
SOLID_MAX_OPEN_DIRECTIONS = 2
# Marching-cubes level on the blurred solid. Above 0.5 takes back part of the walls' one-cell
# thickness; a uniform erosion instead would delete thin parts such as a figure's base.
SOLID_LEVEL = 0.75


@dataclass
class PrintReport:
    faces: int = 0
    vertices: int = 0
    watertight: bool = False
    winding_consistent: bool = False
    size_mm: list[float] = field(default_factory=list)
    volume_cm3: float | None = None
    parts_removed: int = 0
    holes_closed: bool = False
    remeshed: bool = False
    flattened: bool = False
    nonmanifold_edges: int = 0
    warnings: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def load_geometry(path: str | Path) -> trimesh.Trimesh:
    """Load any mesh/scene file as a single geometry-only mesh with welded vertices."""
    loaded = trimesh.load(str(path), force="mesh", process=False)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )
    # Textured exports split vertices along UV seams; welding them back is what makes
    # the surface closed again.
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _rotation(axis: str, degrees: float) -> np.ndarray:
    direction = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    return trimesh.transformations.rotation_matrix(np.radians(degrees), direction)


def orient(mesh: trimesh.Trimesh, settings: PrintSettings) -> None:
    if settings.source_up == "y":
        # +90 deg about X maps +Y to +Z.
        mesh.apply_transform(_rotation("x", 90))
    elif settings.source_up != "z":
        raise ValueError(f"source_up must be 'y' or 'z', got {settings.source_up!r}")
    if settings.rotate_x_deg:
        mesh.apply_transform(_rotation("x", settings.rotate_x_deg))
    if settings.rotate_z_deg:
        mesh.apply_transform(_rotation("z", settings.rotate_z_deg))


def remove_small_parts(mesh: trimesh.Trimesh, min_ratio: float) -> tuple[trimesh.Trimesh, int]:
    if min_ratio <= 0:
        return mesh, 0
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    counts = np.bincount(labels)
    keep = np.flatnonzero(counts >= counts.max() * min_ratio)
    if len(keep) == len(counts):
        return mesh, 0
    mask = np.isin(labels, keep)
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return mesh, int(len(counts) - len(keep))


def _to_meshset(mesh: trimesh.Trimesh) -> pymeshlab.MeshSet:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    ))
    return ms


def _from_meshset(ms: pymeshlab.MeshSet) -> trimesh.Trimesh:
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)


def nonmanifold_edge_count(mesh: trimesh.Trimesh) -> int:
    """Edges shared by more than two faces."""
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return int((counts > 2).sum())


def repair(mesh: trimesh.Trimesh, max_hole_edges: int, report: PrintReport) -> trimesh.Trimesh:
    nonmanifold = nonmanifold_edge_count(mesh)
    if nonmanifold:
        # Meshlab's non-manifold repair deletes faces and scrambles the (consistent) winding
        # of such meshes; solidify() handles them.
        log.info("%d non-manifold edges: surface repair skipped", nonmanifold)
        return mesh
    ms = _to_meshset(mesh)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_unreferenced_vertices()
    try:
        ms.meshing_close_holes(maxholesize=max_hole_edges, selfintersection=False)
        report.holes_closed = True
    except pymeshlab.PyMeshLabException as exc:
        report.warnings.append(f"hole closing failed: {exc}")
    try:
        ms.meshing_re_orient_faces_coherently()
    except pymeshlab.PyMeshLabException as exc:
        report.warnings.append(f"face re-orientation failed: {exc}")
    ms.meshing_remove_unreferenced_vertices()
    out = _from_meshset(ms)
    trimesh.repair.fix_normals(out)
    return out


def scale_and_place(mesh: trimesh.Trimesh, target_height: float | None) -> None:
    """Scale so the Z extent equals target_height, then sit the model on z=0 centred in XY."""
    if target_height:
        height = float(mesh.extents[2])
        if height <= 0:
            raise ValueError("mesh has zero height")
        mesh.apply_scale(target_height / height)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])


def _trim_solid(mesh: trimesh.Trimesh, height: float) -> trimesh.Trimesh:
    """Keep what is above `height`. Unlike slice_plane's cap, this keeps a closed mesh closed."""
    solid = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
    ))
    out = solid.trim_by_plane((0.0, 0.0, 1.0), height).to_mesh()
    return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=False)


def flatten_base(mesh: trimesh.Trimesh, depth: float, report: PrintReport) -> trimesh.Trimesh:
    """Cut `depth` off the bottom (mesh must already sit on z=0) and cap the cut."""
    try:
        if mesh.is_volume:
            cut = _trim_solid(mesh, depth)
        else:
            cut = mesh.slice_plane(plane_origin=[0, 0, depth], plane_normal=[0, 0, 1], cap=True)
    except Exception as exc:  # trimesh raises a variety of errors on bad cross-sections
        report.warnings.append(f"base flattening failed: {exc}")
        return mesh
    if cut is None or len(cut.faces) == 0:
        report.warnings.append("base flattening removed the whole mesh; skipped")
        return mesh
    cut.apply_translation([0, 0, -depth])
    report.flattened = True
    return cut


def _mark_surface(mesh: trimesh.Trimesh, origin: np.ndarray, cell: float, wall: np.ndarray) -> None:
    """Set every cell of `wall` that the surface passes through (points at most half a cell apart)."""
    tris = mesh.triangles
    sides = tris - np.roll(tris, 1, axis=1)
    longest = np.sqrt((sides ** 2).sum(axis=2)).max(axis=1)
    steps = np.ceil(longest / (cell * 0.5)).astype(int).clip(1)
    for n in np.unique(steps):
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        inside = i + j <= n
        a, b = i[inside] / n, j[inside] / n
        c = 1 - a - b
        chosen = tris[steps == n]
        for start in range(0, len(chosen), max(1, 2_000_000 // len(a))):
            part = chosen[start:start + max(1, 2_000_000 // len(a))]
            points = part[:, None, 0] * a[None, :, None] + part[:, None, 1] * b[None, :, None]                 + part[:, None, 2] * c[None, :, None]
            index = ((points.reshape(-1, 3) - origin) / cell).astype(int)
            wall[index[:, 0], index[:, 1], index[:, 2]] = True


def _blocked_along(wall: np.ndarray, step: tuple[int, int, int]) -> np.ndarray:
    """True where a ray from the cell along `step` hits `wall`, sweeping back from the far side."""
    sweep = next(axis for axis in range(3) if step[axis])
    du, dv = (step[axis] for axis in range(3) if axis != sweep)
    walls = np.moveaxis(wall, sweep, 0)
    blocked = np.zeros(walls.shape, dtype=bool)
    layers = range(len(walls) - 2, -1, -1) if step[sweep] > 0 else range(1, len(walls))
    for layer in layers:
        ahead = layer + step[sweep]
        seen = walls[ahead] | blocked[ahead]
        h, w = seen.shape
        # Cell (u, v) of this layer looks at (u + du, v + dv) of the layer ahead.
        blocked[layer][max(-du, 0):h + min(-du, 0), max(-dv, 0):w + min(-dv, 0)] =             seen[max(du, 0):h + min(du, 0), max(dv, 0):w + min(dv, 0)]
    return np.moveaxis(blocked, 0, sweep)


def _enclosed_cells(wall: np.ndarray) -> np.ndarray:
    """Cells that are not `wall` and see out in almost none of the 26 grid directions.

    A cell that sees out both ways along any line is in a through-gap (between the legs, in
    the loop of an arm on a hip), which stays open however narrow it is. Only cavities and
    the bottoms of deep blind pits qualify, and a deep blind pit would only trap resin.
    """
    open_count = np.zeros(wall.shape, dtype=np.uint8)
    through = np.zeros(wall.shape, dtype=bool)
    for step in itertools.product((-1, 0, 1), repeat=3):
        if step > (0, 0, 0):  # one of each opposite pair
            forward = ~_blocked_along(wall, step)
            backward = ~_blocked_along(wall, tuple(-c for c in step))
            open_count += forward.astype(np.uint8) + backward
            through |= forward & backward
    return (open_count <= SOLID_MAX_OPEN_DIRECTIONS) & ~through & ~wall


def solidify(mesh: trimesh.Trimesh, cell: float) -> trimesh.Trimesh:
    """Rebuild any surface as a closed solid, ignoring its winding and closure.

    An open, non-manifold surface wound both ways has no inside or outside, so nothing
    signed (repair, signed distance, winding numbers) works.
    Here the surface is only a set of voxel walls. A cell behind a hole sees out in a few
    directions and a real concavity in many, which tells the two apart ("ray stabbing",
    Nooruddin & Turk 2003). The interior found that way plugs the holes, then a flood fill
    from outside finds the true exterior at full resolution; the rest is the solid.
    """
    cell = max(cell, float(mesh.extents.max()) / SOLID_MAX_CELLS)
    margin = 2 * SOLID_COARSE_FACTOR
    shape = (np.ceil(mesh.extents / cell / SOLID_COARSE_FACTOR).astype(int) + 5) * SOLID_COARSE_FACTOR
    origin = mesh.bounds[0] - margin * cell
    wall = np.zeros(shape, dtype=bool)
    _mark_surface(mesh, origin, cell, wall)

    f = SOLID_COARSE_FACTOR
    coarse = wall.reshape(shape[0] // f, f, shape[1] // f, f, shape[2] // f, f).any(axis=(1, 3, 5))
    interior = _enclosed_cells(coarse).repeat(f, 0).repeat(f, 1).repeat(f, 2)

    # Grow the coarse interior up to the walls only, so it fills the gap behind the surface
    # and plugs holes without bulging through them.
    thick_wall = ndimage.binary_dilation(wall)
    plug = ndimage.binary_dilation(interior, iterations=f + 1, mask=~thick_wall)
    labels, _ = ndimage.label(~(thick_wall | plug))
    solid = labels != labels[0, 0, 0]
    del labels, plug, thick_wall, interior, wall

    field_ = ndimage.gaussian_filter(solid.astype(np.float32), 1.0)
    vertices, faces, _, _ = marching_cubes(field_, level=SOLID_LEVEL, spacing=(cell, cell, cell))
    out = trimesh.Trimesh(vertices + origin + cell / 2, faces, process=False)  # samples are cell centres
    if out.volume < 0:
        out.invert()
    return out


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if not target_faces or len(mesh.faces) <= target_faces:
        return mesh
    ms = _to_meshset(mesh)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        qualitythr=0.5,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=True,
        autoclean=True,
    )
    out = _from_meshset(ms)
    if out.is_watertight:  # fix_normals guesses wrong on open/non-manifold meshes
        trimesh.repair.fix_normals(out)
    return out


def preview_copy(mesh: trimesh.Trimesh, whole: bool = False) -> trimesh.Trimesh:
    """Y-up copy normalised to unit height, which browser viewers frame correctly.

    `whole` normalises the longest side instead, for models much wider than they are tall.
    """
    preview = mesh.copy()
    preview.apply_transform(_rotation("x", -90))
    size = float(preview.extents.max() if whole else preview.extents[1]) or 1.0
    preview.apply_scale(1.0 / size)
    preview.apply_translation(-preview.bounds.mean(axis=0))
    return preview


def make_preview(src: str | Path, dst: str | Path, source_up: str = "y", max_faces: int = PREVIEW_FACES) -> Path:
    """Write a reduced browser copy of a model's mesh, untouched by print prep."""
    mesh = load_geometry(src)
    orient(mesh, PrintSettings(source_up=source_up))
    if len(mesh.faces) > max_faces:
        # 14 s on UltraShape's 9.5M faces, where decimate() takes 190 s; a preview does not
        # need its topology guarantees.
        points, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32),
            target_count=max_faces,
        )
        mesh = trimesh.Trimesh(points, faces, process=False)
    preview_copy(mesh).export(str(dst))
    return Path(dst)


def prepare(src: str | Path, out_stem: str | Path, settings: PrintSettings | None = None) -> PrintReport:
    """Run the full print-prep chain on `src` and write `<out_stem>.<fmt>` files."""
    settings = settings or PrintSettings()
    report = PrintReport()
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_geometry(src)
    orient(mesh, settings)
    mesh = repair(mesh, settings.close_holes_max_edges, report)
    # Scale with the flatten allowance included so the final height still hits the target.
    height = settings.target_height_mm
    if height and settings.flatten_base_mm > 0:
        height += settings.flatten_base_mm
    scale_and_place(mesh, height)  # solidify() works in millimetres

    # Before flattening: capping the cut needs a closed cross-section.
    if settings.remesh_if_open and not mesh.is_watertight:
        mesh = solidify(mesh, settings.remesh_cell_mm)
        report.remeshed = True
    # After closing: an open surface can be in thousands of fragments, so before that a real
    # part (such as a pick's head) can look like a floater.
    mesh, report.parts_removed = remove_small_parts(mesh, settings.min_component_ratio)
    scale_and_place(mesh, height)

    if settings.flatten_base_mm > 0:
        mesh = flatten_base(mesh, settings.flatten_base_mm, report)

    target = settings.target_faces or (REMESH_FACE_CAP if report.remeshed else 0)
    mesh = decimate(mesh, target)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()  # inside-out shell

    report.faces = int(len(mesh.faces))
    report.vertices = int(len(mesh.vertices))
    report.watertight = bool(mesh.is_watertight)
    report.winding_consistent = bool(mesh.is_winding_consistent)
    report.nonmanifold_edges = nonmanifold_edge_count(mesh)
    report.size_mm = [round(float(v), 3) for v in mesh.extents]
    if report.watertight:
        report.volume_cm3 = round(abs(float(mesh.volume)) / 1000.0, 3)
        # Solid figures fill ~10% of their bounding box; double-walled shells ~1%.
        fill = abs(float(mesh.volume)) / float(np.prod(mesh.extents))
        if fill < HOLLOW_FILL_RATIO:
            report.warnings.append(
                f"volume is only {fill:.1%} of the bounding box: the mesh may be a hollow shell"
            )
    else:
        report.warnings.append("mesh is not watertight; your slicer's repair may be needed")

    for fmt in settings.formats:
        path = out_stem.with_suffix(f".{fmt}")
        mesh.export(str(path))
        report.files[fmt] = str(path)
    preview_path = out_stem.with_name(out_stem.name + "_preview.glb")
    preview_copy(mesh, settings.preview_whole).export(str(preview_path))
    report.files["preview"] = str(preview_path)
    return report
