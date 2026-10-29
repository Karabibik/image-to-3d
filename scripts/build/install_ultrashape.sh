#!/usr/bin/env bash
# UltraShape 1.0 refiner (inference only).
source "$(dirname "$0")/lib.sh"

REPO=https://github.com/PKU-YuanGroup/UltraShape-1.0
SHA=5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66
CUBVH_SHA=de1badf89b54975d407c42a02a47d41fa149e486

clone_pinned "$REPO" "$SHA" "$MODELS/UltraShape-1.0"
make_venv ultrashape python3.10
vpip ultrashape -r "$ENVS/ultrashape.txt"
# CUDA extension, compiled against the shared torch, hence no build isolation.
# --no-deps: its runtime only needs numpy/torch; the declared kiui dependency is unused.
vpip ultrashape --no-build-isolation --no-deps "git+https://github.com/ashawkey/cubvh@$CUBVH_SHA"
check_import ultrashape "$MODELS/UltraShape-1.0" \
    "import sys; sys.path.insert(0, 'scripts'); import cubvh, flash_attn; import infer_dit_refine; from ultrashape.pipelines import UltraShapePipeline"
