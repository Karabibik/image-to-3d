"""Pixal3D pixel-aligned image-to-3D, single-view and multi-view (turnaround sheets).

Runs in /opt/venvs/pixal3d with the upstream repo at MODEL_ROOT. Calls the upstream
inference.run_inference / inference_mv.run_inference so camera estimation (MoGe-2), view
loading and GLB export stay identical; only pipeline loading is cached between jobs.
"""

import json
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

# Appended, not prepended, so nothing in app/ can shadow an upstream module.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
from multiview import camera_to_world  # noqa: E402  (numpy + PIL only)

MODEL_ID = "TencentARC/Pixal3D"
# Camera rig of the upstream multi-view example: 20 degree fov, distance 3.1192.
MV_FOV = 0.3490658503988659
MV_DISTANCE = 3.1192049980163574

_pipelines: dict = {}


def _load(kind: str, low_vram: bool, loader):
    key = (kind, low_vram)
    if key not in _pipelines:
        import torch

        _pipelines.clear()  # one pipeline in VRAM at a time
        torch.cuda.empty_cache()
        log(f"loading {MODEL_ID} {kind} (low_vram={low_vram}; first run downloads the weights)")
        _pipelines[key] = loader()
    return _pipelines[key]


_upstream_init = pixal.init_pipeline


def _cached_init(model_path=MODEL_ID, device="cuda", low_vram=False):
    # Skips the multi-view denoisers (~22 GB) the single-view pipeline never loads.
    return _load("single-view", low_vram, lambda: _upstream_init(
        patched_pipeline_dir(MODEL_ID, ignore=["*_mv.*"]), device=device, low_vram=low_vram))


pixal.init_pipeline = _cached_init
_pixal_mv = None


def _mv_module():
    global _pixal_mv
    if _pixal_mv is None:
        import inference_mv

        upstream_mv_init = inference_mv.init_pipeline

        def cached_mv_init(model_path=MODEL_ID, config_file="pipeline_mv.json", device="cuda", low_vram=False):
            return _load("multi-view", low_vram, lambda: upstream_mv_init(
                patched_pipeline_dir(MODEL_ID, config=config_file, allow=["ckpts/*_dec_*", "ckpts/*_mv.*"]),
                config_file, device=device, low_vram=low_vram))

        inference_mv.init_pipeline = cached_mv_init
        _pixal_mv = inference_mv
    return _pixal_mv


def _write_transforms(views: dict, folder: str, elevation: float) -> str:
    """Make `folder` the dataset-style directory inference_mv reads (front view first)."""
    frames = [
        {
            "file_path": os.path.relpath(views[slot], folder),
            "name": slot,
            "transform_matrix": camera_to_world(slot, MV_DISTANCE, elevation),
        }
        for slot in sorted(views, key=lambda slot: slot != "front")
    ]
    with open(os.path.join(folder, "transforms.json"), "w") as f:
        json.dump({"camera_angle_x": MV_FOV, "mesh_scale": 1.0, "frames": frames}, f, indent=2)
    return folder


def generate(job: dict) -> dict:
    params = job.get("params", {})
    if job.get("views"):
        # The mesh comes out in the front camera's frame, so with elevated views it is tilted
        # by the elevation; print prep levels it (generator level_param in models.yaml).
        folder = os.path.dirname(os.path.abspath(job["out"]))
        _mv_module().run_inference(
            views_dir=_write_transforms(job["views"], folder, float(params.get("elevation", 0.0))),
            output_path=job["out"],
            seed=int(job.get("seed", 0)),
            ss_sampling_steps=int(params.get("ss_steps", 12)),
            shape_slat_sampling_steps=int(params.get("shape_steps", 12)),
            low_vram=bool(params.get("low_vram", False)),
            resolution=int(params.get("resolution", 1536)),
        )
        return {"mesh": job["out"], "info": {"views": sorted(job["views"]), **cuda_memory_info()}}
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
