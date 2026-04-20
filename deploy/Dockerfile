# TextWorld VR — deploy image
# Serves a Ray Serve graph that generates layered 3DGS scenes from text prompts.
#
# Build:   docker build -t textworld-vr:latest -f deploy/Dockerfile .
# Run:     docker run --rm --gpus all -p 8000:8000 -p 8080:8080 \
#              -v $PWD/models:/models -v $PWD/scenes:/scenes \
#              -e HF_TOKEN=hf_... textworld-vr:latest
#
# Dual-conda design mirrors the Zaratan cluster:
#   /opt/conda/envs/main  — Python 3.12, torch+cu121, gsplat, diff-gaussian-rasterization
#   /opt/conda/envs/lp3d  — Python 3.9, torch+cu118, 360monodepth, LayerPano3D deps
# The Ray Serve deployment dispatches each stage to whichever env it needs.

ARG CUDA_VERSION=12.1.1
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
# Re-declare in-stage so ${TORCH_CUDA_ARCH_LIST} is visible here. Without this the
# outer ARG is global-only (pre-FROM) and the ENV on the next line resolves empty,
# which makes torch.cpp_extension raise IndexError when building gsplat kernels.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# System deps for OpenCV, ffmpeg, nginx, git, build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl wget ca-certificates build-essential cmake \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      ffmpeg libeigen3-dev libopencv-dev \
      nginx tini \
    && rm -rf /var/lib/apt/lists/*

# Conda
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh -o /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p $CONDA_DIR && rm /tmp/mc.sh \
    && $CONDA_DIR/bin/conda config --set always_yes yes --set changeps1 no \
    && $CONDA_DIR/bin/conda config --add channels conda-forge
ENV PATH=$CONDA_DIR/bin:$PATH

# ---------- main env (serving / rendering / novel-view eval) ----------
COPY textworld-vr/deploy/env/main.yml /tmp/main.yml
RUN conda env create -f /tmp/main.yml && conda clean -afy

# ---------- lp3d env (360monodepth + LayerPano3D scripts) ----------
COPY textworld-vr/deploy/env/lp3d.yml /tmp/lp3d.yml
RUN conda env create -f /tmp/lp3d.yml && conda clean -afy

# LayerPano3D source (needed at runtime for gen_panodepth, run_layerpano, etc.)
ENV LP3D_ROOT=/opt/LayerPano3D
RUN git clone --depth 1 https://github.com/YS-IMTech/LayerPano3D.git ${LP3D_ROOT}

# Build 360monodepth's C extension inside lp3d env
WORKDIR ${LP3D_ROOT}/submodules/360monodepth
RUN if [ -f code/cpp/CMakeLists.txt ]; then \
        /opt/conda/envs/lp3d/bin/pip install pybind11 && \
        cd code/cpp && mkdir -p build && cd build && \
        cmake -DCMAKE_PREFIX_PATH=/opt/conda/envs/lp3d .. && \
        make -j4 && cp *.so ../../python/src/utility/ || true ; \
    fi

# Build diff-gaussian-rasterization + simple-knn in main env for cu121 arches.
# --no-build-isolation is required: their setup.py imports torch at import time
# to discover the CUDA ABI, and PEP 517 build isolation hides the installed torch.
# We also install build-deps first so setuptools/wheel/ninja are ready.
WORKDIR /tmp/build
ENV PIP_NO_CACHE_DIR=1
RUN /opt/conda/envs/main/bin/pip install --upgrade pip setuptools wheel ninja && \
    /opt/conda/envs/main/bin/pip install --no-build-isolation \
        git+https://github.com/graphdeco-inria/gaussian-splatting.git#subdirectory=submodules/diff-gaussian-rasterization \
        git+https://github.com/camenduru/simple-knn.git

# Application code
WORKDIR /app

COPY textworld-vr/stage1_panorama ./stage1_panorama
COPY textworld-vr/stage2_multiview ./stage2_multiview
COPY textworld-vr/stage3_3dgs ./stage3_3dgs
COPY textworld-vr/evaluate.py ./evaluate.py
COPY textworld-vr/configs ./configs
COPY textworld-vr/scripts/zaratan/parallax_proof.py ./tools/parallax_proof.py
COPY textworld-vr/scripts/zaratan/eval_novel_view.py ./tools/eval_novel_view.py
COPY textworld-vr/scripts/zaratan/merge_layered_plys.py ./tools/merge_layered_plys.py
COPY textworld-vr/scripts/zaratan/depth_layer_fallback.py ./tools/depth_layer_fallback.py
COPY textworld-vr/deploy/server ./server
COPY textworld-vr/deploy/render_worker ./render_worker
COPY textworld-vr/deploy/viewer /srv/viewer
COPY textworld-vr/deploy/nginx.conf /etc/nginx/nginx.conf
COPY textworld-vr/deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model + scene volumes (bind-mount at runtime)
ENV HF_HOME=/models/hf_cache
ENV SCENES_DIR=/scenes
ENV LP3D_CHECKPOINTS=${LP3D_ROOT}/checkpoints
RUN mkdir -p /models /scenes ${LP3D_CHECKPOINTS}

EXPOSE 8000 8080 8265
VOLUME ["/models", "/scenes"]

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
