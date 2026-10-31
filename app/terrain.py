"""Layered worlds: merge their meshes for viewing and cut printable terrain tiles out of them.

A world is a set of meshes around a camera at the origin, Z-up, with the sky as a far shell.
Its depth is only known up to scale, so nothing here is in metres: ground distances are in
camera heights (the camera to the ground below it), and a tile gets millimetres when it is cut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree
from trimesh.ray.ray_pyembree import RayMeshIntersector

from .printprep import PrintReport, PrintSettings, prepare, preview_copy

SKY = "sky"
Y_UP = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
# Ground within this angle of straight down tells how high the camera stood.
NADIR_CONE_DEG = 10
# The map ends where this share of the ground is in view. Panorama pixels are even per angle, so
# that is about 13 camera heights out; beyond it the ground is a few stretched pixels.
MAP_QUANTILE = 0.95
MAP_PIXELS = 1024
# Far ground has gaps between its vertices; wider gaps than this (in pixels) are really empty.
MAP_FILL_PIXELS = 8
MAP_BACKGROUND = (38, 38, 43)
# Triangles reaching into a tile from further out than this share of its size are missed.
CROP_MARGIN = 0.25
WORLD_PREVIEW_FACES = 600_000
# Less of a tile than this hit real ground; the rest is filled in, which deserves a warning.
MIN_COVERAGE = 0.7


@dataclass
class TileSettings:
    # Position and size on the ground, in camera heights.
    center_x: float = 0.0
    center_y: float = 0.0
    size: float = 4.0
    rotation_deg: float = 0.0
    # Height samples along one side.
    grid: int = 400
    # Printed size.
    width_mm: float = 50.0
    base_mm: float = 3.0
    # Vertical stretch of the terrain, and a blur over the height samples (in samples).
    exaggeration: float = 1.0
    smoothing: float = 0.7


def load_ground(layers: list[dict]) -> trimesh.Trimesh:
    """Every layer but the sky as one mesh, colours kept."""
    meshes = [trimesh.load(str(layer["path"]), process=False) for layer in layers if layer["type"] != SKY]
    if not meshes:
        raise ValueError("This world has no ground layers.")
    return trimesh.util.concatenate(meshes)


def camera_height(vertices: np.ndarray) -> float:
    below = vertices[vertices[:, 2] < 0]
    if not len(below):
        raise ValueError("This world has no ground below the camera.")
    reach = -below[:, 2] * math.tan(math.radians(NADIR_CONE_DEG))
    cone = below[np.hypot(below[:, 0], below[:, 1]) < reach]
    return float(np.median(-(cone if len(cone) else below)[:, 2]))


def top_view(vertices: np.ndarray, colors: np.ndarray) -> tuple[Image.Image, float]:
    """A straight-down picture of the ground. Takes camera heights; also returns its half width."""
    half = float(np.quantile(np.hypot(vertices[:, 0], vertices[:, 1]), MAP_QUANTILE))
    scale = MAP_PIXELS / (2 * half)
    column = ((vertices[:, 0] + half) * scale).astype(np.int64)
    row = ((half - vertices[:, 1]) * scale).astype(np.int64)
    inside = np.flatnonzero((column >= 0) & (column < MAP_PIXELS) & (row >= 0) & (row < MAP_PIXELS))
    order = inside[np.argsort(vertices[inside, 2])]  # lowest first: the highest point of a pixel is written last

    picture = np.zeros((MAP_PIXELS, MAP_PIXELS, 3), dtype=np.uint8)
    seen = np.zeros((MAP_PIXELS, MAP_PIXELS), dtype=bool)
    picture[row[order], column[order]] = colors[order, :3]
    seen[row[order], column[order]] = True
    gaps, nearest = ndimage.distance_transform_edt(~seen, return_indices=True)
    picture = picture[nearest[0], nearest[1]]
    picture[gaps > MAP_FILL_PIXELS] = MAP_BACKGROUND
    return Image.fromarray(picture), half


def _axes(settings: TileSettings) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(settings.rotation_deg)
    return np.array([math.cos(angle), math.sin(angle)]), np.array([-math.sin(angle), math.cos(angle)])


def grid_step(half: float) -> int:
    """Spacing of the map's grid lines, in camera heights."""
    return next((step for step in (1, 2, 5, 10, 20, 50) if 2 * half / step <= 14), 100)


def map_with_tile(view: Image.Image, half: float, settings: TileSettings) -> Image.Image:
    """The top view with a grid, a cross where the camera stood and the outline of the tile."""
    image = view.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    scale = image.width / (2 * half)

    def pixel(x: float, y: float) -> tuple[float, float]:
        return (x + half) * scale, (half - y) * scale

    step = grid_step(half)
    for line in np.arange(-(half // step) * step, half, step):
        draw.line([pixel(line, -half), pixel(line, half)], fill=(255, 255, 255, 45))
        draw.line([pixel(-half, line), pixel(half, line)], fill=(255, 255, 255, 45))
    x, y = pixel(0, 0)
    draw.line([(x - 12, y), (x + 12, y)], fill=(255, 255, 255, 220), width=2)
    draw.line([(x, y - 12), (x, y + 12)], fill=(255, 255, 255, 220), width=2)

    axis_u, axis_v = _axes(settings)
    center = np.array([settings.center_x, settings.center_y])
    corners = [center + (u * axis_u + v * axis_v) * settings.size / 2 for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    outline = [pixel(*corner) for corner in corners]
    draw.polygon(outline, fill=(255, 196, 60, 40))
    draw.line([*outline, outline[0]], fill=(255, 196, 60, 255), width=4)
    return image


def height_grid(ground: trimesh.Trimesh, settings: TileSettings) -> tuple[np.ndarray, float]:
    """The ground under a tile as rows of heights, and the share of them that hit real ground.

    Each sample is the highest surface there, so the sky must not be in `ground`. Ground is in
    camera heights. Rows run along the tile's own v axis, columns along u.
    """
    axis_u, axis_v = _axes(settings)
    center = np.array([settings.center_x, settings.center_y])
    vertices = np.asarray(ground.vertices)
    faces = np.asarray(ground.faces)

    # The ray test only needs what is near; a world can have tens of millions of faces.
    relative = vertices[:, :2] - center
    reach = settings.size * (0.5 + CROP_MARGIN)
    near = (np.abs(relative @ axis_u) < reach) & (np.abs(relative @ axis_v) < reach)
    crop = trimesh.Trimesh(vertices, faces[near[faces].any(axis=1)], process=False)
    if not len(crop.faces):
        raise ValueError("There is no ground inside this tile: pick a spot on the map.")
    crop.remove_unreferenced_vertices()

    steps = np.linspace(-settings.size / 2, settings.size / 2, settings.grid)
    u, v = np.meshgrid(steps, steps)
    points = center + u[..., None] * axis_u + v[..., None] * axis_v
    above = crop.bounds[1, 2] + 1.0
    origins = np.column_stack([points.reshape(-1, 2), np.full(settings.grid ** 2, above)])
    down = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    hits, rays, _ = RayMeshIntersector(crop).intersects_location(origins, down, multiple_hits=False)

    heights = np.full(settings.grid ** 2, np.nan)
    heights[rays] = hits[:, 2]
    heights = heights.reshape(settings.grid, settings.grid)
    empty = np.isnan(heights)
    if empty.all():
        raise ValueError("There is no ground inside this tile: pick a spot on the map.")
    nearest = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    heights = heights[nearest[0], nearest[1]]
    if settings.smoothing > 0:
        heights = ndimage.gaussian_filter(heights, settings.smoothing, mode="nearest")
    return heights, float(1 - empty.mean())


def solid_tile(heights: np.ndarray, width_mm: float, base_mm: float) -> trimesh.Trimesh:
    """A closed solid by construction: heights in mm on top, four walls, a flat bottom at z=0."""
    n = len(heights)
    xs = np.linspace(0.0, width_mm, n)
    x, y = np.meshgrid(xs, xs)
    top = np.column_stack([x.ravel(), y.ravel(), (heights + base_mm).ravel()])
    index = np.arange(n * n).reshape(n, n)

    a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    upper = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])

    # The rim of the top, counter-clockwise from above, and the same points on the bottom.
    rim = np.concatenate([index[0, :-1], index[:-1, -1], index[-1, :0:-1], index[:0:-1, 0]])
    low = n * n + np.arange(len(rim))
    middle = n * n + len(rim)
    bottom = top[rim] * [1, 1, 0]
    vertices = np.vstack([top, bottom, [[width_mm / 2, width_mm / 2, 0.0]]])

    rim_next, low_next = np.roll(rim, -1), np.roll(low, -1)
    walls = np.vstack([np.column_stack([low, low_next, rim_next]), np.column_stack([low, rim_next, rim])])
    floor = np.column_stack([np.full(len(low), middle), low_next, low])
    return trimesh.Trimesh(vertices, np.vstack([upper, walls, floor]), process=False)


def cut_tile(layers: list[dict], unit: float, settings: TileSettings, out_stem: Path,
             target_faces: int | None = 200_000, formats: tuple[str, ...] = ("stl", "3mf")) -> PrintReport:
    """Write a print-ready tile. `unit` is the world's camera height in its own units."""
    ground = load_ground(layers)
    ground.apply_scale(1.0 / unit)
    heights, coverage = height_grid(ground, settings)
    mm_per_unit = settings.width_mm / settings.size
    relief = (heights - heights.min()) * settings.exaggeration * mm_per_unit
    solid = solid_tile(relief, settings.width_mm, settings.base_mm)

    source = Path(f"{out_stem}_source.ply")
    solid.export(str(source))
    report = prepare(source, out_stem, PrintSettings(
        source_up="z", target_height_mm=None, flatten_base_mm=0.0, min_component_ratio=0.0, remesh_if_open=False,
        target_faces=target_faces, formats=formats, preview_whole=True,
    ))
    source.unlink()
    if coverage < MIN_COVERAGE:
        report.warnings.append(f"only {coverage:.0%} of this tile hit real ground; the rest is filled in")
    return report


def _world_preview(ground: trimesh.Trimesh, radius: float, path: Path) -> None:
    """A browser copy of the ground within `radius`. Colours come from the nearest original vertex."""
    vertices = np.asarray(ground.vertices)
    near = np.hypot(vertices[:, 0], vertices[:, 1]) < radius
    faces = np.asarray(ground.faces)
    crop = trimesh.Trimesh(vertices, faces[near[faces].all(axis=1)], process=False,
                           vertex_colors=np.asarray(ground.visual.vertex_colors))
    crop.remove_unreferenced_vertices()
    if len(crop.faces) > WORLD_PREVIEW_FACES:
        points, reduced = fast_simplification.simplify(
            np.asarray(crop.vertices, dtype=np.float64), np.asarray(crop.faces, dtype=np.int32),
            target_count=WORLD_PREVIEW_FACES,
        )
        colors = np.asarray(crop.visual.vertex_colors)[cKDTree(crop.vertices).query(points)[1]]
        crop = trimesh.Trimesh(points, reduced, vertex_colors=colors, process=False)
    preview_copy(crop, whole=True).export(str(path))


def export_world(layers: list[dict], folder: Path) -> dict:
    """Write world.glb, its browser preview and the map the Tile step works on."""
    ground = load_ground(layers)
    unit = camera_height(np.asarray(ground.vertices))
    view, half = top_view(np.asarray(ground.vertices) / unit, np.asarray(ground.visual.vertex_colors))
    view.save(folder / "top_view.png")
    _world_preview(ground, half * unit, folder / "world_preview.glb")

    sky = [trimesh.load(str(layer["path"]), process=False) for layer in layers if layer["type"] == SKY]
    merged = trimesh.util.concatenate([ground, *sky])
    merged.apply_transform(Y_UP)
    merged.export(str(folder / "world.glb"))
    return {"unit": unit, "map_half": half, "map": str(folder / "top_view.png"),
            "world": str(folder / "world.glb"), "preview": str(folder / "world_preview.glb")}
