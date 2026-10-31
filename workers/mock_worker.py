"""Stand-in worker for developing the UI without a GPU (MOCK_WORKERS=1).

Speaks the real worker protocol. `generate` writes MOCK_MESH if set (any mesh file),
otherwise a simple figure on a hex base; `refine` copies its input mesh; `paint` wraps
a colour gradient around its input mesh. The world tasks write a gradient panorama and a small
world: a house, ground with a hill, and a sky shell.
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from worker_common import log, serve


def _placeholder_figure() -> trimesh.Trimesh:
    base = trimesh.creation.cylinder(radius=0.5, height=0.1, sections=6)
    body = trimesh.creation.capsule(height=1.0, radius=0.25, count=[32, 32])
    body.apply_translation([0, 0, 0.05 + 0.25 + 0.5])
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.22)
    head.apply_translation([0, 0, 1.75])
    figure = trimesh.boolean.union([base, body, head], engine="manifold")
    # Model outputs are glTF (Y-up); match that.
    figure.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
    return figure


def generate(job: dict) -> dict:
    if job.get("views"):
        log(f"mock {job['model']}: views {', '.join(sorted(job['views']))}")
    for step in range(3):
        log(f"mock {job['model']}: step {step + 1}/3")
        time.sleep(float(os.environ.get("MOCK_DELAY", "0.7")))
    source = os.environ.get("MOCK_MESH")
    if source:
        shutil.copyfile(source, job["out"])
    else:
        _placeholder_figure().export(job["out"])
    return {"mesh": job["out"], "info": {"mock": True}}


def refine(job: dict) -> dict:
    log(f"mock {job['model']}: refining {job['mesh']}")
    time.sleep(float(os.environ.get("MOCK_DELAY", "0.7")))
    shutil.copyfile(job["mesh"], job["out"])
    return {"mesh": job["out"], "info": {"mock": True}}


def paint(job: dict) -> dict:
    log(f"mock {job['model']}: painting {job['mesh']}")
    time.sleep(float(os.environ.get("MOCK_DELAY", "0.7")))
    mesh = trimesh.load(job["mesh"], force="mesh", process=False)
    height = (mesh.vertices[:, 1] - mesh.bounds[0, 1]) / (mesh.extents[1] or 1.0)
    ramp = np.linspace([200, 60, 40], [40, 90, 210], 256).astype(np.uint8)[None]
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.column_stack([height, np.full_like(height, 0.5)]),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(ramp).resize((256, 8)), metallicFactor=0.0, roughnessFactor=0.8,
        ),
    )
    mesh.export(job["out"])
    return {"mesh": job["out"], "info": {"mock": True}}


def panorama(job: dict) -> dict:
    log(f"mock {job['task']}: seed {job['seed']}")
    time.sleep(float(os.environ.get("MOCK_DELAY", "0.7")))
    sky = np.linspace([110, 160, 225], [225, 235, 245], 480)
    ground = np.linspace([120, 150, 90], [60, 90, 50], 480)
    rows = np.vstack([sky, ground]).astype(np.uint8)
    Image.fromarray(np.tile(rows[:, None], (1, 1920, 1))).save(job["out"])
    return {"image": job["out"], "info": {"mock": True}}


def _mock_ground() -> trimesh.Trimesh:
    cells = 300
    steps = np.linspace(-30, 30, cells)
    x, y = np.meshgrid(steps, steps)
    z = -1.7 + 2.5 * np.exp(-((x - 9) ** 2 + (y - 6) ** 2) / 12) + 0.15 * np.sin(x / 2) * np.cos(y / 3)
    index = np.arange(cells * cells).reshape(cells, cells)
    a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    shade = z.ravel()
    colors = np.column_stack([70 + 30 * shade, 130 + 25 * shade, 60 + 10 * shade, np.full(shade.size, 255)])
    return trimesh.Trimesh(
        np.column_stack([x.ravel(), y.ravel(), shade]),
        np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])]),
        vertex_colors=colors.clip(0, 255).astype(np.uint8), process=False,
    )


def scene(job: dict) -> dict:
    log(f"mock scene: {job.get('classes')} {job.get('labels_fg1')} {job.get('labels_fg2')}")
    time.sleep(float(os.environ.get("MOCK_DELAY", "0.7")))
    house = trimesh.creation.box(extents=(3, 3, 3.5))
    house.apply_translation([-7, 5, -1.7 + 1.75])
    house.visual.vertex_colors = [170, 90, 70, 255]
    sky = trimesh.creation.icosphere(subdivisions=4, radius=80.0)
    sky.visual.vertex_colors = [130, 175, 235, 255]
    layers = []
    for kind, mesh in (("fg1", house), ("bg", _mock_ground()), ("sky", sky)):
        file = f"mesh_layer{len(layers)}.ply"
        mesh.export(Path(job["out"]) / file)
        layers.append({"type": kind, "file": file, "faces": len(mesh.faces)})
    return {"layers": layers, "info": {"mock": True}}


if __name__ == "__main__":
    serve({"generate": generate, "refine": refine, "paint": paint, "pano_image": panorama, "pano_text": panorama,
           "scene": scene}, loaded=lambda: ["mock"])
