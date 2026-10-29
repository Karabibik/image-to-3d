#!/usr/bin/env bash
# Hunyuan3D 2.1, shape generation only (the texture pipeline is not installed).
source "$(dirname "$0")/lib.sh"

REPO=https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
SHA=82920d643c0dc2f7bfd7255f45f62d386edfe60c

clone_pinned "$REPO" "$SHA" "$MODELS/Hunyuan3D-2.1"
make_venv hunyuan python3.10
vpip hunyuan -r "$ENVS/hunyuan.txt"
check_import hunyuan "$MODELS/Hunyuan3D-2.1/hy3dshape" \
    "import hy3dshape; from hy3dshape.pipelines import export_to_trimesh; from hy3dshape.rembg import BackgroundRemover"
