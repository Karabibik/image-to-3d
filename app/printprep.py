"""Turn a generated mesh into a print-ready model.

Steps, in order: load geometry only, convert to Z-up, drop floaters, repair and close
holes, scale to a target height in millimetres, optionally flatten the base, optionally
voxel-remesh if the mesh is still open, decimate to a face budget, export STL/3MF plus
a Y-up GLB for the browser preview.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh

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
    # Voxel remesh (in output units) when repair leaves the mesh open. Smooths detail finer than the cell.
    remesh_if_open: bool = False
    remesh_cell_mm: float = 0.05
    # Face budget after decimation; None or 0 keeps every face.
    target_faces: int | None = 200_000
    formats: tuple[str, ...] = ("stl", "3mf")


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


def repair(mesh: trimesh.Trimesh, max_hole_edges: int, report: PrintReport) -> trimesh.Trimesh:
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


def flatten_base(mesh: trimesh.Trimesh, depth: float, report: PrintReport) -> trimesh.Trimesh:
    """Cut `depth` off the bottom (mesh must already sit on z=0) and cap the cut."""
    try:
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


def voxel_remesh(mesh: trimesh.Trimesh, cell: float) -> trimesh.Trimesh:
    ms = _to_meshset(mesh)
    ms.generate_resampled_uniform_mesh(
        cellsize=pymeshlab.PureValue(cell),
        offset=pymeshlab.PureValue(0.0),
        mergeclosevert=True,
        multisample=True,
        absdist=True,
    )
    out = _from_meshset(ms)
    trimesh.repair.fix_normals(out)
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
    trimesh.repair.fix_normals(out)
    return out


def preview_copy(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Y-up copy normalised to unit height, which browser viewers frame correctly."""
    preview = mesh.copy()
    preview.apply_transform(_rotation("x", -90))
    height = float(preview.extents[1]) or 1.0
    preview.apply_scale(1.0 / height)
    preview.apply_translation(-preview.bounds.mean(axis=0))
    return preview


def prepare(src: str | Path, out_stem: str | Path, settings: PrintSettings | None = None) -> PrintReport:
    """Run the full print-prep chain on `src` and write `<out_stem>.<fmt>` files."""
    settings = settings or PrintSettings()
    report = PrintReport()
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_geometry(src)
    orient(mesh, settings)
    mesh, report.parts_removed = remove_small_parts(mesh, settings.min_component_ratio)
    mesh = repair(mesh, settings.close_holes_max_edges, report)
    # Scale with the flatten allowance included so the final height still hits the target.
    height = settings.target_height_mm
    if height and settings.flatten_base_mm > 0:
        height += settings.flatten_base_mm
    scale_and_place(mesh, height)

    if settings.flatten_base_mm > 0:
        mesh = flatten_base(mesh, settings.flatten_base_mm, report)

    if settings.remesh_if_open and not mesh.is_watertight:
        mesh = voxel_remesh(mesh, settings.remesh_cell_mm)
        report.remeshed = True
        scale_and_place(mesh, None)

    mesh = decimate(mesh, settings.target_faces or 0)

    report.faces = int(len(mesh.faces))
    report.vertices = int(len(mesh.vertices))
    report.watertight = bool(mesh.is_watertight)
    report.winding_consistent = bool(mesh.is_winding_consistent)
    report.size_mm = [round(float(v), 3) for v in mesh.extents]
    if report.watertight:
        report.volume_cm3 = round(abs(float(mesh.volume)) / 1000.0, 3)
    else:
        report.warnings.append("mesh is not watertight; check it in your slicer or enable remesh")

    for fmt in settings.formats:
        path = out_stem.with_suffix(f".{fmt}")
        mesh.export(str(path))
        report.files[fmt] = str(path)
    preview_path = out_stem.with_name(out_stem.name + "_preview.glb")
    preview_copy(mesh).export(str(preview_path))
    report.files["preview"] = str(preview_path)
    return report
