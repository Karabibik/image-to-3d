"""UltraShape 1.0 refinement (image + coarse mesh -> detailed mesh).

Runs in /opt/venvs/ultrashape with the upstream repo at MODEL_ROOT. Mirrors
scripts/infer_dit_refine.py, keeping the loaded models between jobs.
"""

import os
import sys

ROOT = os.environ.get("MODEL_ROOT", "/opt/models/UltraShape-1.0")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from worker_common import cuda_memory_info, log, serve  # noqa: E402

CONFIG = os.path.join(ROOT, "configs", "infer_dit_refine.yaml")

_cache: dict = {}


def _get_pipeline(low_vram: bool):
    key = ("pipeline", low_vram)
    if key not in _cache:
        from huggingface_hub import hf_hub_download
        from infer_dit_refine import load_models
        from ultrashape.pipelines import UltraShapePipeline

        _cache.clear()
        torch.cuda.empty_cache()
        ckpt = hf_hub_download("infinith/UltraShape", "ultrashape_v1.pt")
        log(f"loading UltraShape from {ckpt} (low_vram={low_vram})")
        components, config = load_models(CONFIG, ckpt, torch.device("cuda"))
        pipeline = UltraShapePipeline(
            vae=components["vae"],
            model=components["dit"],
            scheduler=components["scheduler"],
            conditioner=components["conditioner"],
            image_processor=components["image_processor"],
        )
        if low_vram:
            pipeline.enable_model_cpu_offload()
        _cache[key] = (pipeline, config)
    return _cache[key]


def refine(job: dict) -> dict:
    from ultrashape.rembg import BackgroundRemover
    from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
    from ultrashape.utils import voxelize_from_point

    params = job.get("params", {})
    pipeline, config = _get_pipeline(bool(params.get("low_vram", False)))

    image = Image.open(job["image"])
    if image.mode != "RGBA":
        image = BackgroundRemover()(image)

    loader = SharpEdgeSurfaceLoader(num_sharp_points=204800, num_uniform_points=204800)
    surface = loader(job["mesh"], normalize_scale=float(params.get("scale", 0.99))).to("cuda", dtype=torch.float16)
    voxel_res = config.model.params.vae_config.params.voxel_query_res
    _, voxel_idx = voxelize_from_point(surface[:, :, :3], int(params.get("num_latents", 32768)), resolution=voxel_res)

    generator = torch.Generator("cuda").manual_seed(int(job.get("seed", 0)))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        meshes, _ = pipeline(
            image=image,
            voxel_cond=voxel_idx,
            generator=generator,
            box_v=1.0,
            mc_level=0.0,
            octree_resolution=int(params.get("octree_res", 1024)),
            num_inference_steps=int(params.get("steps", 50)),
            num_chunks=int(params.get("chunk_size", 8000)),
        )
    mesh = meshes[0]
    mesh.export(job["out"])
    return {"mesh": job["out"], "info": {"faces": int(len(mesh.faces)), **cuda_memory_info()}}


if __name__ == "__main__":
    serve({"refine": refine}, loaded=lambda: ["ultrashape"] if _cache else [])
