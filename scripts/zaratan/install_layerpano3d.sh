#!/bin/bash
# install_layerpano3d.sh — best-effort install on Zaratan login node.
#
# Strategy:
#  1. conda create env py=3.9 (separate from our torch 2.5.1 + cu121 main venv)
#  2. conda-forge provides ceres-solver, glog, gflags, suitesparse, pybind11, tbb, cmake
#     (no sudo needed, unlike the README's apt-get recipe)
#  3. module load eigen, boost, opencv (available in Zaratan module tree)
#  4. pip install torch 2.4.0 + cu118 and diffusers/xformers/natten pinned to that wheel
#  5. Compile submodules/diff-gaussian-rasterization + simple-knn
#  6. Try 360monodepth compile — if that specific submodule fails, continue
#     (LayerPano3D's depth can be substituted by our DA-v2 pipeline)
#
# Produces a log at $HOME/scratch/phase4/textworld-vr/logs/lp3d_install.log

set -uo pipefail

REPO="$HOME/scratch/phase4/textworld-vr/shared/LayerPano3D"
LOG="$HOME/scratch/phase4/textworld-vr/logs/lp3d_install.log"
CONDA="$HOME/scratch/miniconda3"

mkdir -p "$(dirname "$LOG")"
exec > "$LOG" 2>&1
echo "== [$(date -Is)] starting LayerPano3D install on $(hostname)"

# 1. Ensure conda is initialized
if [ ! -x "$CONDA/bin/conda" ]; then
    echo "ERROR: miniconda not found at $CONDA"
    exit 1
fi
source "$CONDA/etc/profile.d/conda.sh"

# 2. Create / activate env
ENV_NAME="lp3d"
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "== creating conda env $ENV_NAME (python=3.9)"
    conda create -n "$ENV_NAME" python=3.9 -y
fi
conda activate "$ENV_NAME"
python --version

# 3. System-lib replacements via conda-forge (no sudo needed)
echo "== installing conda-forge build deps (ceres, glog, pybind11, tbb, cmake, eigen, opencv, suitesparse)"
conda install -c conda-forge -y \
    ceres-solver glog gflags suitesparse pybind11 tbb cmake \
    eigen libopencv opencv boost-cpp ninja 2>&1 | tail -20

# 4. PyTorch cu118 pinned
echo "== installing torch 2.4.0 + cu118"
pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -6

echo "== installing natten, xformers pinned for cu118/torch2.4"
pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -4 || true
pip install --no-cache-dir -f https://shi-labs.com/natten/wheels/cu118/torch240/ natten==0.14.4 2>&1 | tail -4 || true

# 5. LayerPano3D requirements.txt (ignore the torch pin in it since we handled above)
echo "== installing LayerPano3D requirements.txt"
cd "$REPO"
# remove the shapely/mmcv/natten/torch line pin that would downgrade
grep -v -E "^(shapely|mmcv|natten|torch|torchvision|xformers)" requirements.txt > /tmp/lp3d_reqs_trimmed.txt || true
pip install -r /tmp/lp3d_reqs_trimmed.txt 2>&1 | tail -20 || true
pip install timm==0.4.12 --no-deps 2>&1 | tail -4 || true

# 6. Compile the easy CUDA submodules (don't need sudo)
echo "== compiling diff-gaussian-rasterization"
pip install -e submodules/diff-gaussian-rasterization 2>&1 | tail -20 || \
    echo "!! diff-gaussian-rasterization failed (might need cuda/12.3 module loaded)"

echo "== compiling simple-knn"
pip install -e submodules/simple-knn 2>&1 | tail -20 || \
    echo "!! simple-knn failed"

# 7. 360monodepth — the hard part. Try, but don't fail the whole install if it breaks.
echo "== attempting 360monodepth compilation"
if [ -d submodules/360monodepth ]; then
    cd submodules/360monodepth
    if [ ! -d code/cpp/3rd_party/pybind11 ]; then
        (cd code/cpp/3rd_party && git clone https://github.com/pybind/pybind11.git && \
         cd pybind11 && mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX="$CONDA/envs/$ENV_NAME" .. && make -j8 && make install) 2>&1 | tail -5 || \
         echo "!! pybind11 build failed"
    fi
    # skip ceres (we have it via conda-forge at $CONDA/envs/$ENV_NAME)
    cd code/cpp
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH="$CONDA/envs/$ENV_NAME" \
          -DEIGEN3_INCLUDE_DIR="$CONDA/envs/$ENV_NAME/include/eigen3" \
          .. 2>&1 | tail -15 || echo "!! cmake failed"
    make -j8 2>&1 | tail -15 || echo "!! make failed"
    cd ../python
    if ls dist/*.whl >/dev/null 2>&1; then
        pip install dist/*.whl 2>&1 | tail -5 || echo "!! wheel install failed"
    else
        echo "!! no wheel built; 360monodepth unavailable. Will substitute DA-v2 at inference."
    fi
    cd "$REPO"
fi

# 8. Pre-stage the LayerPano3D checkpoints (FLUX LoRA + Lama)
echo "== pre-staging LayerPano3D checkpoints"
mkdir -p checkpoints
CKPT_URL_LORA="https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA/resolve/main/lora_hubs/pano_lora_720%2A1440_v1.safetensors"
CKPT_URL_LAMA="https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth"
if [ ! -f checkpoints/pano_lora_720x1440_v1.safetensors ]; then
    curl -L -o checkpoints/pano_lora_720x1440_v1.safetensors "$CKPT_URL_LORA" 2>&1 | tail -3 || \
        echo "!! LoRA download failed"
fi
if [ ! -f checkpoints/ControlNetLama.pth ]; then
    curl -L -o checkpoints/ControlNetLama.pth "$CKPT_URL_LAMA" 2>&1 | tail -3 || \
        echo "!! Lama download failed"
fi

# 9. Sanity: can we import the key modules?
echo "== import smoke"
python - <<PY 2>&1 | tail -30
import sys
for m in ["torch", "diffusers", "xformers", "natten", "PIL", "einops"]:
    try:
        mod = __import__(m)
        v = getattr(mod, "__version__", "?")
        print(f"ok  {m:20s} {v}")
    except Exception as e:
        print(f"!!  {m:20s} {type(e).__name__}: {e}")
try:
    import torch
    print(f"cuda available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"!!  cuda check: {e}")
PY

echo "== [$(date -Is)] install script finished"
