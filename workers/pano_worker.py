"""HY-Pano 2.0 on its Qwen-Image-Edit backend (one image -> 360-degree panorama).

Runs in /opt/venvs/hypano with the upstream repo at MODEL_ROOT.
"""

import os
import sys

ROOT = os.environ.get("MODEL_ROOT", "/opt/models/HY-World-2.0")
sys.path.insert(0, os.path.join(ROOT, "hyworld2", "panogen"))

from worker_common import cuda_memory_info, log, serve  # noqa: E402

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline_with_qwen_image import HunyuanPanoPipeline

        log("loading HY-Pano 2.0 (first run downloads ~55 GB)")
        _pipeline = HunyuanPanoPipeline.from_pretrained()
    return _pipeline


def pano_image(job: dict) -> dict:
    params = job.get("params", {})
    # Upstream's default size: 1952 less its 32 px edge blend is the 1920 x 960 HunyuanWorld expects.
    panorama = _get_pipeline()(job["image"], prompt=job.get("prompt", ""), seed=job["seed"],
                               num_inference_steps=int(params.get("steps", 40)))
    panorama.save(job["out"])
    return {"image": job["out"], "info": cuda_memory_info()}


if __name__ == "__main__":
    serve({"pano_image": pano_image}, loaded=lambda: ["hypano"] if _pipeline else [])
