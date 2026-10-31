#!/usr/bin/env bash
# Torch stack of the world image, installed once into the system Python 3.10 and inherited by
# its model venvs (--system-site-packages): torch 2.7.1/cu128, the version HY-Pano 2.0 is
# tested on. CUDA 12.8 is what brings Blackwell (RTX PRO 6000) support.
source "$(dirname "$0")/lib.sh"

PY=python3.10

$PY -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
# hf_transfer because the image enables HF_HUB_ENABLE_HF_TRANSFER and every venv downloads weights.
$PY -m pip install --no-cache-dir packaging wheel setuptools hf_transfer
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
