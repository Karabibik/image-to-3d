#!/usr/bin/env bash
# Hunyuan3D-2mv multi-view shape generation. Runs in the hunyuan venv (Hunyuan3D 2.1's pins
# cover hy3dgen's shape path), so this only adds the upstream code. No texture pipeline.
source "$(dirname "$0")/lib.sh"

REPO=https://github.com/Tencent-Hunyuan/Hunyuan3D-2
SHA=f8db63096c8282cb27354314d896feba5ba6ff8a

clone_pinned "$REPO" "$SHA" "$MODELS/Hunyuan3D-2"
# Keep the package and its licence; the rest is demo assets, docs and the Blender addon.
find "$MODELS/Hunyuan3D-2" -mindepth 1 -maxdepth 1 \
    ! -name hy3dgen ! -name LICENSE ! -name NOTICE ! -name .pinned -exec rm -rf {} +
check_import hunyuan "$MODELS/Hunyuan3D-2" \
    "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; from hy3dgen.shapegen.preprocessors import MVImageProcessorV2; import huggingface_hub"
