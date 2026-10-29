"""Hi3D v1.0 geometry refinement (image + coarse mesh -> high-res mesh). Experimental.

Runs in /opt/venvs/hi3d with the upstream package installed from MODEL_ROOT.
Output is geometry only, recentred and normalised to unit radius (print prep rescales).
"""

import os

import trimesh
from PIL import Image

from worker_common import cuda_memory_info, log, serve

MODEL_ID = "hi3d-official/Hi3Dv1.0"

_pipelines: dict = {}


def _progress(event) -> None:
    log(f"pass {event.pass_index + 1}/{event.num_passes} step {event.step}/{event.num_inference_steps}")


def _get_pipeline(cpu_offload: bool):
    if cpu_offload not in _pipelines:
        import torch
        from sparc3d import Sparc3DPipeline

        _pipelines.clear()
        torch.cuda.empty_cache()
        log(f"loading {MODEL_ID} (cpu_offload={cpu_offload})")
        pipeline = Sparc3DPipeline.from_pretrained(MODEL_ID, progress_callback=_progress)
        if cpu_offload:
            pipeline.enable_sequential_cpu_offload("cuda")
        else:
            pipeline = pipeline.to("cuda")
        _pipelines[cpu_offload] = pipeline
    return _pipelines[cpu_offload]


def refine(job: dict) -> dict:
    params = job.get("params", {})
    pipeline = _get_pipeline(bool(params.get("cpu_offload", False)))
    result = pipeline(
        Image.open(job["image"]),
        trimesh.load(job["mesh"], force="mesh", process=False),
        resolution=int(params.get("resolution", 1024)),
        num_inference_steps=int(params.get("steps", 25)),
        seed=int(job.get("seed", 0)),
    )
    result.mesh.export(job["out"])
    return {
        "mesh": job["out"],
        "info": {"resolution": result.resolution, "faces": int(len(result.mesh.faces)), **cuda_memory_info()},
    }


if __name__ == "__main__":
    os.chdir(os.environ.get("MODEL_ROOT", "/opt/models/Hi3Dv1.0"))
    serve({"refine": refine}, loaded=lambda: ["hi3d"] if _pipelines else [])
