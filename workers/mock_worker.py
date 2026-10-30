"""Stand-in worker for developing the UI without a GPU (MOCK_WORKERS=1).

Speaks the real worker protocol. `generate` writes MOCK_MESH if set (any mesh file),
otherwise a simple figure on a hex base; `refine` copies its input mesh.
"""

import os
import shutil
import time

import numpy as np
import trimesh

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


if __name__ == "__main__":
    serve({"generate": generate, "refine": refine}, loaded=lambda: ["mock"])
