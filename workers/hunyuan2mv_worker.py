"""Hunyuan3D-2mv multi-view shape generation (front/left/back/right views -> untextured mesh).

Runs in /opt/venvs/hunyuan with the upstream Hunyuan3D-2 repo at MODEL_ROOT. Mirrors
examples/shape_gen_multiview.py and fast_shape_gen_multiview.py (turbo + FlashVDM).
"""

import os
import sys

ROOT = os.environ.get("MODEL_ROOT", "/opt/models/Hunyuan3D-2")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from PIL import Image  # noqa: E402

from worker_common import cuda_memory_info, log, serve  # noqa: E402

MODEL_ID = "tencent/Hunyuan3D-2mv"
SUBFOLDERS = {
    "standard": "hunyuan3d-dit-v2-mv",
    "turbo": "hunyuan3d-dit-v2-mv-turbo",
    "fast": "hunyuan3d-dit-v2-mv-fast",
}
# enable_flashvdm() swaps in this VAE for the turbo variant.
TURBO_VAE = ("tencent/Hunyuan3D-2", "hunyuan3d-vae-v2-0-turbo")

_pipelines: dict = {}


def _fetch(repo: str, subfolder: str) -> None:
    # hy3dgen looks in $HY3DGEN_MODELS/<repo>/<subfolder> before falling back to the HF cache.
    base = os.path.expanduser(os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen"))
    snapshot_download(
        repo, local_dir=os.path.join(base, repo),
        allow_patterns=[f"{subfolder}/config.yaml", f"{subfolder}/model.fp16.safetensors"],
    )


def _get_pipeline(variant: str):
    if variant not in _pipelines:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        _pipelines.clear()
        torch.cuda.empty_cache()
        subfolder = SUBFOLDERS[variant]
        log(f"loading {MODEL_ID}/{subfolder} (first run downloads ~5 GB)")
        _fetch(MODEL_ID, subfolder)
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID, subfolder=subfolder, variant="fp16", use_safetensors=True, device="cuda"
        )
        if variant == "turbo":
            _fetch(*TURBO_VAE)
            pipeline.enable_flashvdm()
        _pipelines[variant] = pipeline
    return _pipelines[variant]


def generate(job: dict) -> dict:
    params = job.get("params", {})
    # The views arrive cut out (RGBA); the pipeline's MVImageProcessorV2 recentres each one.
    views = {slot: Image.open(path).convert("RGBA") for slot, path in job["views"].items()}
    mesh = _get_pipeline(params.get("variant", "standard"))(
        image=views,
        num_inference_steps=int(params.get("steps", 50)),
        guidance_scale=float(params.get("guidance_scale", 5.0)),
        octree_resolution=int(params.get("octree_resolution", 384)),
        num_chunks=int(params.get("num_chunks", 200000)),
        generator=torch.Generator().manual_seed(int(job.get("seed", 0))),
        output_type="trimesh",
    )[0]
    mesh.export(job["out"])
    return {"mesh": job["out"], "info": {"faces": int(len(mesh.faces)), "views": sorted(views), **cuda_memory_info()}}


if __name__ == "__main__":
    serve({"generate": generate}, loaded=lambda: [f"hunyuan2mv:{v}" for v in _pipelines])
