# syntax=docker/dockerfile:1.7
#
# Image-to-miniature RunPod pod image.
#
#   /opt/venvs/<model>  one venv per model family (Hunyuan3D 2.1, UltraShape, TRELLIS.2,
#                       Pixal3D, Hi3D), layered over two shared torch stacks
#   /opt/models/<repo>  upstream code, pinned to exact commits by scripts/build/*.sh
#   /opt/app            this repo's UI (port 7860) and model workers (127.0.0.1:7871-7875)
#   /workspace          pod volume: HF weights cache, outputs, logs (nothing baked here)
#
# Model weights are not baked in; they download on first use or via PREFETCH_MODELS.
# Build: docker build --platform linux/amd64 -f docker/Dockerfile -t <user>/image-to-3d:<tag> .

# CUDA 12.4.1 cudnn-devel on Ubuntu 22.04 with RunPod's /start.sh (SSH, Jupyter, nginx),
# python3.10-3.13. Devel toolkit is needed to compile the models' CUDA extensions.
FROM runpod/base:0.7.0-ubuntu2204-cuda1241

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# A40/A6000/3090 (8.6), A100 (8.0), L40S/4090 (8.9), H100 (9.0). Blackwell needs a cu128 build.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ARG MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    U2NET_HOME=/workspace/.cache/rembg \
    DATA_DIR=/workspace

# Fail fast if the base image stops providing what the rest of the build assumes.
RUN python3.10 --version && python3.11 --version && nvcc --version | tail -1 \
    && test -x /start.sh && grep -q post_start /start.sh

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libopengl0 libglu1-mesa libegl1 libgles2 \
        libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libxrender1 libxi6 ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/build/lib.sh scripts/build/install_torch.sh /opt/build/
RUN bash /opt/build/install_torch.sh py310
RUN bash /opt/build/install_torch.sh py311

# One layer per model family, so changing one pin only rebuilds that layer onwards.
COPY envs/hunyuan.txt /opt/build/envs/
COPY scripts/build/install_hunyuan.sh /opt/build/
RUN bash /opt/build/install_hunyuan.sh

COPY envs/ultrashape.txt /opt/build/envs/
COPY scripts/build/install_ultrashape.sh /opt/build/
RUN bash /opt/build/install_ultrashape.sh

COPY envs/trellis.txt envs/pixal3d.txt /opt/build/envs/
COPY scripts/build/install_trellis.sh /opt/build/
RUN bash /opt/build/install_trellis.sh

COPY scripts/build/install_hi3d.sh /opt/build/
RUN bash /opt/build/install_hi3d.sh

# UI / orchestrator venv: no torch, no system site-packages.
COPY app/requirements.txt /opt/build/envs/ui.txt
RUN python3.11 -m venv /opt/venvs/ui && /opt/venvs/ui/bin/python -m pip install -r /opt/build/envs/ui.txt

COPY app /opt/app/app
COPY workers /opt/app/workers
COPY tests /opt/app/tests
COPY scripts/smoke_test.py /opt/app/scripts/smoke_test.py
COPY docker/run_ui.sh /opt/app/run_ui.sh
COPY docker/post_start.sh /post_start.sh
RUN chmod +x /opt/app/run_ui.sh /post_start.sh \
    && cd /opt/app && /opt/venvs/ui/bin/python -m pytest -q tests

WORKDIR /opt/app
EXPOSE 7860
# CMD is inherited: the base /start.sh brings up SSH/Jupyter, then runs /post_start.sh,
# which starts the UI under a restart loop.
