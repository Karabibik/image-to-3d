#!/usr/bin/env bash
# Hunyuan3D-Paint 2.1 texture pipeline. The code is the hy3dpaint folder of the repo that
# install_hunyuan.sh pins; it gets its own venv so the tested shape venv does not change.
source "$(dirname "$0")/lib.sh"

ROOT=$MODELS/Hunyuan3D-2.1
PAINT=$ROOT/hy3dpaint
ESRGAN_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
ESRGAN_SHA256=4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1

make_venv hunyuan_paint python3.10
vpip hunyuan_paint -r "$ENVS/hunyuan.txt" -r "$ENVS/hunyuan_paint.txt"
# --no-deps: only their inference code runs; the declared dependencies add training and
# face-restoration packages. No build isolation, because basicsr's setup needs torch.
vpip hunyuan_paint --no-build-isolation --no-deps basicsr==1.4.2 realesrgan==0.3.0

# CUDA extension, compiled against the shared torch, hence no build isolation.
vpip hunyuan_paint --no-build-isolation --no-deps "$PAINT/custom_rasterizer"

# Upstream's compile_mesh_painter.sh, except that it asks the venv's Python rather than
# whichever python3-config comes first for the extension suffix.
python=$VENVS/hunyuan_paint/bin/python
suffix=$("$python" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# shellcheck disable=SC2046  # the include flags must split into words
(cd "$PAINT/DifferentiableRenderer" && c++ -O3 -Wall -shared -std=c++11 -fPIC \
    $("$python" -m pybind11 --includes) mesh_inpaint_processor.cpp -o "mesh_inpaint_processor$suffix")

mkdir -p "$PAINT/ckpt"
curl -fsSL -o "$PAINT/ckpt/RealESRGAN_x4plus.pth" "$ESRGAN_URL"
echo "$ESRGAN_SHA256  $PAINT/ckpt/RealESRGAN_x4plus.pth" | sha256sum -c -

check_dists hunyuan_paint custom_rasterizer xatlas
check_import hunyuan_paint "$ROOT" "
import sys, types
sys.path[:0] = ['.', 'hy3dpaint']
sys.modules['bpy'] = types.ModuleType('bpy')
from torchvision_fix import apply_fix
apply_fix()
import custom_rasterizer
from DifferentiableRenderer.mesh_inpaint_processor import meshVerticeInpaint
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
from convert_utils import create_glb_with_pbr_materials
"
