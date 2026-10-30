#!/usr/bin/env bash
# Shared torch stack, installed once into the system Python 3.10 and inherited by the model
# venvs (--system-site-packages): torch 2.5.1/cu124, the version Hunyuan3D 2.1, UltraShape
# and Hi3D are tested on.
source "$(dirname "$0")/lib.sh"

PY=python3.10
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abi{ABI}-cp310-cp310-linux_x86_64.whl"

$PY -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

ABI=$($PY -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
$PY -m pip install --no-cache-dir "${FA_WHEEL/\{ABI\}/$ABI}"
# Build tooling for the CUDA extensions, plus hf_transfer because the base image
# enables HF_HUB_ENABLE_HF_TRANSFER and every venv downloads weights.
$PY -m pip install --no-cache-dir ninja packaging wheel setuptools hf_transfer
$PY -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"
