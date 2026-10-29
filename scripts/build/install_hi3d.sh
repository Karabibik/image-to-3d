#!/usr/bin/env bash
# Hi3D v1.0 refiner (experimental).
source "$(dirname "$0")/lib.sh"

REPO=https://github.com/hi3d-official/Hi3Dv1.0
SHA=13657ad09afaa92e350b461a1cd66399e44a59c4

clone_pinned "$REPO" "$SHA" "$MODELS/Hi3Dv1.0"
make_venv hi3d python3.10
# Upstream's verified stack, minus the torch/flash-attn pins the shared base already
# satisfies (torch 2.5.1 instead of their measured 2.5.0; same flash-attn build).
grep -vE '^(torch|torchvision|flash-attn)==' "$MODELS/Hi3Dv1.0/docker/constraints.txt" > /tmp/hi3d-constraints.txt
vpip hi3d -c /tmp/hi3d-constraints.txt -e "$MODELS/Hi3Dv1.0[inference]"
check_import hi3d "$MODELS/Hi3Dv1.0" "import spconv.pytorch, flash_attn, pyfqmr"
check_import hi3d "$MODELS/Hi3Dv1.0" "from sparc3d import Sparc3DPipeline" soft
