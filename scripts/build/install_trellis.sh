#!/usr/bin/env bash
# TRELLIS.2 and Pixal3D. They share the CUDA extensions (built once as wheels) but get
# separate venvs because they need different utils3d builds.
source "$(dirname "$0")/lib.sh"

TRELLIS_REPO=https://github.com/microsoft/TRELLIS.2
TRELLIS_SHA=75fbf0183001ed9876c8dbb35de6b68552ee08bd
PIXAL_REPO=https://github.com/TencentARC/Pixal3D
PIXAL_SHA=f7cf38429b0bd264f1995f0f8743a88b1c728b94
# Extension pins, matching TRELLIS.2's setup.sh sources.
NVDIFFRAST="git+https://github.com/NVlabs/nvdiffrast@253ac4fcea7de5f396371124af597e6cc957bfae"     # v0.4.0
NVDIFFREC="git+https://github.com/JeffreyXiang/nvdiffrec@b296927cc7fd01c2ac1087c8065c4d7248f72da4"  # renderutils
CUMESH="git+https://github.com/JeffreyXiang/CuMesh@12289e1062f0603f2f0d0771b02e1395d247f26f"
FLEXGEMM="git+https://github.com/JeffreyXiang/FlexGEMM@6dd94a859c26ee8246888502eada3dd8ad85532e"
UTILS3D_TRELLIS="git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
UTILS3D_PIXAL="https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"
# Last MoGe commit before MoGe 3 (which drags in its own FlexGEMM build); has MoGe-2.
MOGE="git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8"
# The torch270cu126 build links the pre-C++11 string ABI and fails to load against the
# CXX11-ABI torch 2.7.0+cu126 wheel; the cu128 build matches it (checked on an A40).
NATTEN="natten==0.21.0+torch270cu128"

WHEELS=/opt/wheels/trellis
clone_pinned "$TRELLIS_REPO" "$TRELLIS_SHA" "$MODELS/TRELLIS.2"
clone_pinned "$PIXAL_REPO" "$PIXAL_SHA" "$MODELS/Pixal3D"

mkdir -p "$WHEELS"
for src in "$NVDIFFRAST" "$NVDIFFREC" "$CUMESH" "$FLEXGEMM" "$MODELS/TRELLIS.2/o-voxel"; do
    python3.11 -m pip wheel --no-cache-dir --no-build-isolation --no-deps -w "$WHEELS" "$src"
done

make_venv trellis python3.11
vpip trellis -r "$ENVS/trellis.txt"
# --no-deps: o-voxel names cumesh/flex_gemm as git URLs that pip won't match to these
# wheels; their runtime deps are listed in trellis.txt instead.
vpip trellis --no-deps "$WHEELS"/*.whl
vpip trellis "$UTILS3D_TRELLIS"
# Distribution names of the extension wheels (the part of each filename before the first "-").
EXT_DISTS=$(for w in "$WHEELS"/*.whl; do basename "$w" | cut -d- -f1; done)
# shellcheck disable=SC2086  # word-split the list on purpose
check_dists trellis $EXT_DISTS utils3d
# flex_gemm builds Triton autotuners at import, which needs a GPU: soft checks only.
check_import trellis "$MODELS/TRELLIS.2" "import o_voxel, cumesh, flex_gemm, nvdiffrast.torch, utils3d" soft
check_import trellis "$MODELS/TRELLIS.2" "from trellis2.pipelines import Trellis2ImageTo3DPipeline" soft

make_venv pixal3d python3.11
vpip pixal3d -r "$ENVS/trellis.txt" -r "$ENVS/pixal3d.txt"
vpip pixal3d --no-deps "$MOGE"
vpip pixal3d --no-deps "$WHEELS"/*.whl
vpip pixal3d "$NATTEN" -f https://whl.natten.org
vpip pixal3d "$UTILS3D_PIXAL"
# shellcheck disable=SC2086
check_dists pixal3d $EXT_DISTS natten moge utils3d
check_import pixal3d "$MODELS/Pixal3D" "import o_voxel, natten, moge, utils3d" soft
check_import pixal3d "$MODELS/Pixal3D" "import inference" soft

rm -rf "$WHEELS"
