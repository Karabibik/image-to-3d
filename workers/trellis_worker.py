"""TRELLIS.2 image-to-3D.

Runs in /opt/venvs/trellis with the upstream repo at MODEL_ROOT. Mirrors the upstream
app.py up to the sampling; the export keeps only the geometry (see worker_mesh).
"""

import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.environ.get("MODEL_ROOT", "/opt/models/TRELLIS.2")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PIL import Image  # noqa: E402

from worker_common import cuda_memory_info, log, serve  # noqa: E402
from worker_hf import patched_pipeline_dir  # noqa: E402
from worker_mesh import geometry_to_glb  # noqa: E402

MODEL_ID = "microsoft/TRELLIS.2-4B"
PIPELINE_TYPES = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        log(f"loading {MODEL_ID} (first run downloads ~16 GB)")
        _pipeline = Trellis2ImageTo3DPipeline.from_pretrained(patched_pipeline_dir(MODEL_ID))
        _pipeline.cuda()
    return _pipeline


def _sampler(params: dict, prefix: str, defaults: dict) -> dict:
    return {key: type(value)(params.get(f"{prefix}_{key}", value)) for key, value in defaults.items()}


def generate(job: dict) -> dict:
    params = job.get("params", {})
    pipeline = _get_pipeline()
    resolution = str(params.get("resolution", "1024"))
    outputs, _ = pipeline.run(
        Image.open(job["image"]),
        seed=int(job.get("seed", 0)),
        sparse_structure_sampler_params=_sampler(params, "ss", {"steps": 12, "guidance_strength": 7.5}),
        shape_slat_sampler_params=_sampler(params, "shape", {"steps": 12, "guidance_strength": 7.5}),
        tex_slat_sampler_params=_sampler(params, "tex", {"steps": 12}),
        pipeline_type=PIPELINE_TYPES[resolution],
        return_latent=True,
    )
    mesh = outputs[0]
    mesh.simplify(16777216)  # nvdiffrast limit, as upstream
    glb = geometry_to_glb(mesh.vertices, mesh.faces, int(params.get("decimation_target", 1000000)))
    glb.export(job["out"])
    return {"mesh": job["out"], "info": {"resolution": resolution, **cuda_memory_info()}}


if __name__ == "__main__":
    serve({"generate": generate}, loaded=lambda: ["trellis"] if _pipeline else [])
