#!/usr/bin/env bash
# Shared torch stacks, installed once into the system interpreters and inherited by the
# model venvs (--system-site-packages):
#   py310: torch 2.5.1/cu124 — Hunyuan3D 2.1, UltraShape, Hi3D (their tested versions)
#   py311: torch 2.7.0/cu126 — TRELLIS.2, Pixal3D (Pixal3D's NATTEN needs torch >= 2.7)
source "$(dirname "$0")/lib.sh"

FA=https://github.com/Dao-AILab/flash-attention/releases/download

case "${1:?usage: install_torch.sh py310|py311}" in
    py310)
        PY=python3.10
        $PY -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu124
        FA_WHEEL="$FA/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abi{ABI}-cp310-cp310-linux_x86_64.whl"
        ;;
    py311)
        PY=python3.11
        $PY -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 \
            --index-url https://download.pytorch.org/whl/cu126
        FA_WHEEL="$FA/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abi{ABI}-cp311-cp311-linux_x86_64.whl"
        ;;
    *) echo "unknown stack $1" >&2; exit 1 ;;
esac

ABI=$($PY -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
$PY -m pip install --no-cache-dir "${FA_WHEEL/\{ABI\}/$ABI}"
# Build tooling for the CUDA extensions, plus hf_transfer because the base image
# enables HF_HUB_ENABLE_HF_TRANSFER and every venv downloads weights.
$PY -m pip install --no-cache-dir ninja packaging wheel setuptools hf_transfer
$PY -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"
