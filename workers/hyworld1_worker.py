"""HunyuanWorld 1.0: panoramas on FLUX and layered world meshes.

Runs in /opt/venvs/hyworld1 with the upstream repo at MODEL_ROOT, on upstream's own demo
classes. The FLUX repos are gated, so HF_TOKEN must belong to an account that accepted them.
"""

import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = os.environ.get("MODEL_ROOT", "/opt/models/HunyuanWorld-1.0")
HOME = Path(os.environ.get("HYWORLD_HOME", "/workspace/.cache/hyworld"))
sys.path.insert(0, ROOT)
HOME.mkdir(parents=True, exist_ok=True)
os.chdir(HOME)  # LayerDecomposition reads ./ZIM/zim_vit_l_2092

import torch  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError  # noqa: E402

from worker_common import cuda_memory_info, log, serve  # noqa: E402

FLUX_DEV = "black-forest-labs/FLUX.1-dev"
FLUX_FILL = "black-forest-labs/FLUX.1-Fill-dev"

_loaded = (None, None)


def _release() -> None:
    global _loaded
    _loaded = (None, None)
    gc.collect()
    torch.cuda.empty_cache()


def _demo(key: tuple, build):
    """Keep one panorama pipeline between jobs; each holds its own copy of FLUX."""
    global _loaded
    if _loaded[0] != key:
        _release()
        log(f"loading {key[0]} (first run downloads ~32 GB)")
        _loaded = (key, build())
    return _loaded[1]


def _need_access(repo: str) -> None:
    """Say what is wrong up front: diffusers reports a refused repo as a model that cannot be found."""
    try:
        HfApi().auth_check(repo)
    except GatedRepoError as exc:
        raise RuntimeError(f"No access to {repo}: accept its license with the account behind the hf_token secret") from exc


def _options(params: dict) -> SimpleNamespace:
    # The fp8 options are for 24 GB cards and need sageattention, which this image leaves out.
    return SimpleNamespace(cache=bool(params.get("cache", False)), fp8_attention=False, fp8_gemm=False)


def pano_text(job: dict) -> dict:
    from demo_panogen import Text2PanoramaDemo

    options = _options(job.get("params", {}))
    _need_access(FLUX_DEV)
    demo = _demo(("text to panorama", options.cache), lambda: Text2PanoramaDemo(options))
    demo.run(job["prompt"], None, job["seed"], os.path.dirname(job["out"]))  # always writes panorama.png
    return {"image": job["out"], "info": cuda_memory_info()}


def pano_image(job: dict) -> dict:
    from demo_panogen import Image2PanoramaDemo

    options = _options(job.get("params", {}))
    _need_access(FLUX_FILL)
    demo = _demo(("image to panorama", options.cache), lambda: Image2PanoramaDemo(options))
    demo.run(job.get("prompt", ""), "", job["image"], job["seed"], os.path.dirname(job["out"]))
    return {"image": job["out"], "info": cuda_memory_info()}


def scene(job: dict) -> dict:
    from demo_scenegen import HYworldDemo

    _release()
    _need_access(FLUX_FILL)
    snapshot_download("naver-iv/zim-anything-vitl", allow_patterns=["zim_vit_l_2092/*"], local_dir=HOME / "ZIM")
    # A new one per job: the sky step moves the models off the GPU when it finishes.
    log("loading the world models (first run downloads ~35 GB)")
    demo = HYworldDemo(_options(job.get("params", {})), seed=job["seed"])
    demo.run(image_path=job["image"], labels_fg1=job.get("labels_fg1", []), labels_fg2=job.get("labels_fg2", []),
             classes=job.get("classes", "outdoor"), output_dir=job["out"])
    layers = [
        {"type": layer["type"], "file": f"mesh_layer{index}.ply", "faces": len(layer["mesh"].triangles)}
        for index, layer in enumerate(demo.hy3d_world.layered_world_mesh)
    ]
    info = cuda_memory_info()
    del demo
    _release()
    return {"layers": layers, "info": info}


if __name__ == "__main__":
    serve({"pano_text": pano_text, "pano_image": pano_image, "scene": scene},
          loaded=lambda: [_loaded[0][0]] if _loaded[0] else [])
