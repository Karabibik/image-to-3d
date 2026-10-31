#!/usr/bin/env bash
# HunyuanWorld 1.0 (FLUX panoramas, layered world meshes). Upstream is a conda environment on
# torch 2.5.0/cu124; this keeps its Python package pins on the cu128 stack. It gets its own venv
# because its diffusers, transformers and numpy conflict with HY-Pano 2.0's.
source "$(dirname "$0")/lib.sh"

ROOT=$MODELS/HunyuanWorld-1.0
ESRGAN_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
ESRGAN_SHA256=49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb

clone_pinned https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 57fa9f3a79eae1079c279968cdb9d82c75fa7c86 "$ROOT"
make_venv hyworld1 python3.10
vpip hyworld1 -r "$ENVS/hyworld1.txt"

# --no-deps: only inference code runs; the declared dependencies add training, face restoration
# and a demo UI. basicsr-fixed is the sdist upstream asks for, and its setup needs torch.
vpip hyworld1 --no-build-isolation --no-deps basicsr-fixed==1.4.2 realesrgan==0.3.0

# Upstream installs MoGe unpinned, but imports moge.model.v1 and utils3d.numpy, which MoGe 3
# dropped. This is the last commit before HunyuanWorld's release, with the utils3d it pins.
clone_pinned https://github.com/EasternJournalist/utils3d c5daf6f6c244d251f252102d09e9b7bcef791a38 /tmp/utils3d
clone_pinned https://github.com/microsoft/MoGe 72fdee98935d93b69f7ce0a1369cfd6e2573fc2f /tmp/moge
clone_pinned https://github.com/naver-ai/ZIM 392e613c8a3d395ce4dda3b51edb6dabc1fd0004 /tmp/zim
vpip hyworld1 --no-deps /tmp/utils3d /tmp/moge /tmp/zim
rm -rf /tmp/utils3d /tmp/moge /tmp/zim

# sr_utils looks here first and would otherwise download it on every new pod.
mkdir -p "$ROOT/hy3dworld/utils/weights"
curl -fsSL -o "$ROOT/hy3dworld/utils/weights/RealESRGAN_x2plus.pth" "$ESRGAN_URL"
echo "$ESRGAN_SHA256  $ROOT/hy3dworld/utils/weights/RealESRGAN_x2plus.pth" | sha256sum -c -

check_dists hyworld1 diffusers basicsr-fixed realesrgan moge utils3d zim_anything open3d
check_import hyworld1 "$ROOT" "
from hy3dworld import Image2PanoramaPipelines, LayerDecomposition, Perspective, Text2PanoramaPipelines, WorldComposer
from demo_panogen import Image2PanoramaDemo, Text2PanoramaDemo
from demo_scenegen import HYworldDemo
"
