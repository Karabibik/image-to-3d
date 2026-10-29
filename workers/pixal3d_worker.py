"""Pixal3D pixel-aligned image-to-3D.

Runs in /opt/venvs/pixal3d with the upstream repo at MODEL_ROOT. Calls the upstream
inference.run_inference so camera estimation (MoGe-2) and GLB export stay identical;
only pipeline loading is cached between jobs.
"""

import os
import sys

ROOT = os.environ.get("MODEL_ROOT", "/opt/models/Pixal3D")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import inference as pixal  # noqa: E402  (upstream inference.py; sets its own env defaults)
import o_voxel  # noqa: E402

from worker_common import cuda_memory_info, log, serve  # noqa: E402

_upstream_to_glb = o_voxel.postprocess.to_glb


def _to_glb_single_layer(*args, **kwargs):
    # inference.py exports with remesh=True, whose narrow-band rebuild wraps every surface
    # in an inner and outer skin (a hollow shell); keep the generated single-layer surface.
    kwargs["remesh"] = False
    return _upstream_to_glb(*args, **kwargs)


o_voxel.postprocess.to_glb = _to_glb_single_layer
from worker_hf import patched_pipeline_dir  # noqa: E402

MODEL_ID = "TencentARC/Pixal3D"

_pipelines: dict = {}
_upstream_init = pixal.init_pipeline


def _cached_init(model_path=MODEL_ID, device="cuda", low_vram=False):
    if low_vram not in _pipelines:
        import torch

        _pipelines.clear()
        torch.cuda.empty_cache()
        log(f"loading {MODEL_ID} (low_vram={low_vram}; first run downloads the weights)")
        _pipelines[low_vram] = _upstream_init(patched_pipeline_dir(MODEL_ID), device=device, low_vram=low_vram)
    return _pipelines[low_vram]


pixal.init_pipeline = _cached_init


def generate(job: dict) -> dict:
    params = job.get("params", {})
    pixal.run_inference(
        image_path=job["image"],
        output_path=job["out"],
        seed=int(job.get("seed", 0)),
        ss_sampling_steps=int(params.get("ss_steps", 12)),
        ss_guidance_strength=float(params.get("ss_guidance_strength", 7.5)),
        shape_slat_sampling_steps=int(params.get("shape_steps", 12)),
        shape_slat_guidance_strength=float(params.get("shape_guidance_strength", 7.5)),
        manual_fov=float(params.get("fov", -1.0)),
        low_vram=bool(params.get("low_vram", False)),
        resolution=int(params.get("resolution", 1536)),
    )
    return {"mesh": job["out"], "info": cuda_memory_info()}


if __name__ == "__main__":
    serve({"generate": generate}, loaded=lambda: ["pixal3d"] if _pipelines else [])
