"""Hunyuan3D 2.1 shape generation (image -> untextured mesh).

Runs in /opt/venvs/hunyuan with the upstream repo at MODEL_ROOT. Mirrors the shape
path of the upstream gradio_app.py.
"""

import os
import sys

ROOT = os.environ.get("MODEL_ROOT", "/opt/models/Hunyuan3D-2.1")
sys.path.insert(0, os.path.join(ROOT, "hy3dshape"))
os.chdir(ROOT)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from worker_common import cuda_memory_info, log, serve  # noqa: E402

MODEL_ID = "tencent/Hunyuan3D-2.1"
SUBFOLDER = "hunyuan3d-dit-v2-1"

_pipeline = None
_rembg = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline

        log(f"loading {MODEL_ID}/{SUBFOLDER} (first run downloads ~7 GB)")
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID, subfolder=SUBFOLDER, use_safetensors=False, device="cuda"
        )
    return _pipeline


def _load_image(path: str) -> Image.Image:
    global _rembg
    image = Image.open(path)
    if image.mode == "RGBA":
        return image
    if _rembg is None:
        from hy3dshape.rembg import BackgroundRemover

        _rembg = BackgroundRemover()
    return _rembg(image.convert("RGB"))


def generate(job: dict) -> dict:
    from hy3dshape.pipelines import export_to_trimesh

    params = job.get("params", {})
    image = _load_image(job["image"])
    outputs = _get_pipeline()(
        image=image,
        num_inference_steps=int(params.get("steps", 30)),
        guidance_scale=float(params.get("guidance_scale", 5.0)),
        generator=torch.Generator().manual_seed(int(job.get("seed", 0))),
        octree_resolution=int(params.get("octree_resolution", 384)),
        num_chunks=int(params.get("num_chunks", 200000)),
        output_type="mesh",
    )
    mesh = export_to_trimesh(outputs)[0]
    mesh.export(job["out"])
    return {"mesh": job["out"], "info": {"faces": int(len(mesh.faces)), **cuda_memory_info()}}


if __name__ == "__main__":
    serve({"generate": generate}, loaded=lambda: ["hunyuan"] if _pipeline else [])
