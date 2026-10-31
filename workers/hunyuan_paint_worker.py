"""Hunyuan3D-Paint 2.1 (mesh + one reference image -> PBR-textured GLB).

Runs in /opt/venvs/hunyuan_paint with the upstream repo at MODEL_ROOT. Mirrors the texture
path of the upstream gradio_app.py. The model only ever looks at one reference image, so
the back and sides are invented, even for turnaround runs.
"""

import os
import sys
import types

ROOT = os.environ.get("MODEL_ROOT", "/opt/models/Hunyuan3D-2.1")
sys.path[:0] = [ROOT, os.path.join(ROOT, "hy3dpaint")]
os.chdir(ROOT)
# Upstream imports Blender at load time, but only for the GLB export that save_glb=False skips.
sys.modules.setdefault("bpy", types.ModuleType("bpy"))

import torch  # noqa: E402
from torchvision_fix import apply_fix  # noqa: E402

apply_fix()  # basicsr imports a torchvision module that has since moved

from worker_common import cuda_memory_info, log, serve  # noqa: E402

_pipeline = None
_settings = None


def _get_pipeline(views: int, resolution: int):
    global _pipeline, _settings
    if _settings != (views, resolution):
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        _pipeline, _settings = None, None
        torch.cuda.empty_cache()
        log(f"loading Hunyuan3D-Paint for {views} views at {resolution} (first run downloads ~11 GB)")
        config = Hunyuan3DPaintConfig(views, resolution)
        config.realesrgan_ckpt_path = os.path.join(ROOT, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
        config.multiview_cfg_path = os.path.join(ROOT, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
        config.custom_pipeline = os.path.join(ROOT, "hy3dpaint/hunyuanpaintpbr")
        _pipeline, _settings = Hunyuan3DPaintPipeline(config), (views, resolution)
    return _pipeline


def paint(job: dict) -> dict:
    from convert_utils import create_glb_with_pbr_materials

    params = job.get("params", {})
    obj_path = os.path.join(os.path.dirname(job["out"]), "textured.obj")
    pipeline = _get_pipeline(int(params.get("max_num_view", 8)), int(params.get("resolution", 768)))
    # The orchestrator has already reduced the mesh, so upstream's remesh is off.
    pipeline(mesh_path=job["mesh"], image_path=job["image"], output_mesh_path=obj_path,
             use_remesh=False, save_glb=False)
    base = os.path.splitext(obj_path)[0]
    textures = {"albedo": f"{base}.jpg", "metallic": f"{base}_metallic.jpg", "roughness": f"{base}_roughness.jpg"}
    create_glb_with_pbr_materials(obj_path, textures, job["out"])
    return {"mesh": job["out"], "info": cuda_memory_info()}


if __name__ == "__main__":
    serve({"paint": paint}, loaded=lambda: ["hunyuan_paint"] if _pipeline else [])
