#!/usr/bin/env bash
# HY-Pano 2.0 on its Qwen-Image-Edit backend (image -> 360 panorama). Only hyworld2/panogen of
# this repo runs here; the backend needs nothing compiled.
source "$(dirname "$0")/lib.sh"

ROOT=$MODELS/HY-World-2.0

clone_pinned https://github.com/Tencent-Hunyuan/HY-World-2.0 df9988efb87bfc0f4947eb3889411cf957478b06 "$ROOT"
make_venv hypano python3.10
vpip hypano -r "$ENVS/hypano.txt"

check_dists hypano diffusers peft
check_import hypano "$ROOT/hyworld2/panogen" "from pipeline_with_qwen_image import HunyuanPanoPipeline"
