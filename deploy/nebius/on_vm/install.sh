#!/bin/bash
# install.sh — runs ON THE VM. One-time native install.
#
# Replaces the Dockerfile with direct conda envs + system nginx + cloned LP3D.
# Idempotent: re-running skips work that is already done.
#
# After this, run.sh (under systemd) starts uvicorn.

set -euo pipefail

REPO="${REPO:-/mnt/src/textworld-vr}"
LP3D_ROOT="${LP3D_ROOT:-/opt/LayerPano3D}"
LP3D_CHECKPOINTS="${LP3D_CHECKPOINTS:-$LP3D_ROOT/checkpoints}"
HF_HOME="${HF_HOME:-/mnt/models/hf_cache}"
SCENES_DIR="${SCENES_DIR:-/mnt/scenes}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}"

echo "== TextWorld VR native install"
echo "   REPO=$REPO  CONDA_DIR=$CONDA_DIR  LP3D_ROOT=$LP3D_ROOT"
echo "   HF_HOME=$HF_HOME  SCENES_DIR=$SCENES_DIR"

# ---------------------------------------------------------------------------
# 1. apt: build deps + nginx + utilities
# ---------------------------------------------------------------------------
echo "-- apt deps --"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg libeigen3-dev \
    libboost-dev libboost-system-dev libboost-filesystem-dev \
    libboost-program-options-dev libboost-thread-dev libgtest-dev \
    libceres-dev \
    nginx libgoogle-glog-dev pkg-config

# ---------------------------------------------------------------------------
# 2. Miniconda (idempotent)
# ---------------------------------------------------------------------------
if [ ! -d "$CONDA_DIR" ]; then
    echo "-- installing Miniconda to $CONDA_DIR --"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh -o /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$CONDA_DIR"
    rm /tmp/mc.sh
fi
export PATH="$CONDA_DIR/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge

# ---------------------------------------------------------------------------
# 3. Conda envs (idempotent: skip if env exists)
# ---------------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx main; then
    echo "-- creating main env (Python 3.12 + cu121) --"
    conda env create -f "$REPO/deploy/env/main.yml"
else
    echo "-- main env already exists --"
fi

if ! conda env list | awk '{print $1}' | grep -qx lp3d; then
    echo "-- creating lp3d env (Python 3.9 + cu118) --"
    conda env create -f "$REPO/deploy/env/lp3d.yml"
else
    echo "-- lp3d env already exists --"
fi

# ---------------------------------------------------------------------------
# 4. LayerPano3D source clone (idempotent)
# ---------------------------------------------------------------------------
if [ ! -d "$LP3D_ROOT/.git" ]; then
    echo "-- cloning LayerPano3D --"
    sudo mkdir -p "$(dirname "$LP3D_ROOT")"
    sudo git clone --depth 1 https://github.com/YS-IMTech/LayerPano3D.git "$LP3D_ROOT"
    sudo chown -R "$USER:$USER" "$LP3D_ROOT"
else
    echo "-- LayerPano3D already cloned --"
fi
mkdir -p "$LP3D_CHECKPOINTS"

# ---------------------------------------------------------------------------
# 5. 360monodepth C++ extension (idempotent: skip if .so already there)
# ---------------------------------------------------------------------------
MONO_SO_DIR="$LP3D_ROOT/submodules/360monodepth/python/src/utility"
MONO_CPP_DIR="$LP3D_ROOT/submodules/360monodepth/code/cpp"
if [ -f "$MONO_CPP_DIR/CMakeLists.txt" ] && ! ls "$MONO_SO_DIR"/*.so >/dev/null 2>&1; then
    echo "-- building 360monodepth C++ extension --"
    "$CONDA_DIR/envs/lp3d/bin/pip" install pybind11
    pushd "$MONO_CPP_DIR" >/dev/null
    mkdir -p build && cd build
    cmake -DCMAKE_PREFIX_PATH="$CONDA_DIR/envs/lp3d" ..
    make -j"$(nproc)"
    cp ./*.so "$MONO_SO_DIR/" || true
    popd >/dev/null
else
    echo "-- 360monodepth already built (or cmake missing) --"
fi

# ---------------------------------------------------------------------------
# 6. (Removed.) Our pipeline uses gsplat (pre-built wheels), not the
#    diff-gaussian-rasterization+simple-knn combo from upstream LP3D, so we
#    don't compile any CUDA extensions in the main env.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7. Persistent dirs (already created by cloud-init under /mnt; symlink shims
#    so the existing api.py defaults SCENES_DIR=/scenes etc. still work).
# ---------------------------------------------------------------------------
sudo mkdir -p /mnt/models /mnt/scenes /mnt/work
sudo chown -R "$USER:$USER" /mnt/models /mnt/scenes /mnt/work
[ -e /scenes ] || sudo ln -sf /mnt/scenes /scenes
[ -e /models ] || sudo ln -sf /mnt/models /models
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# 8. Stage models (idempotent; skips already-downloaded weights)
# ---------------------------------------------------------------------------
echo "-- staging models (this is slow on first run: ~80 GB from HuggingFace) --"
# HF_TOKEN is read from ~/textworld-vr.env if present.
if [ -f "$HOME/textworld-vr.env" ]; then
    set -a; source "$HOME/textworld-vr.env"; set +a
fi
"$CONDA_DIR/envs/main/bin/python" "$REPO/deploy/server/stage_models.py" \
    --hf-cache "$HF_HOME" --checkpoints "$LP3D_CHECKPOINTS" \
    || { echo "model staging failed"; exit 1; }

# ---------------------------------------------------------------------------
# 9. nginx site (static viewer + scene file server). We use the distro
#    default /etc/nginx/nginx.conf and just drop in a per-site config. The
#    shipped deploy/nginx.conf was written for the docker container and uses
#    /dev/stderr / /dev/stdout / /tmp/nginx.pid which don't work under the
#    distro's systemd unit.
# ---------------------------------------------------------------------------
echo "-- nginx --"
sudo mkdir -p /srv
sudo rm -rf /srv/viewer
sudo cp -r "$REPO/deploy/viewer" /srv/viewer
sudo cp "$REPO/deploy/nebius/on_vm/nginx-textworld-vr.conf" /etc/nginx/sites-available/textworld-vr
sudo rm -f /etc/nginx/sites-enabled/default /etc/nginx/sites-enabled/textworld-vr
sudo ln -s /etc/nginx/sites-available/textworld-vr /etc/nginx/sites-enabled/textworld-vr
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx

# ---------------------------------------------------------------------------
# 10. systemd unit (substituting paths)
# ---------------------------------------------------------------------------
echo "-- systemd unit --"
sudo tee /etc/systemd/system/textworld-vr.service >/dev/null <<EOF
[Unit]
Description=TextWorld VR API (FastAPI + uvicorn)
After=network-online.target nginx.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO
EnvironmentFile=-$HOME/textworld-vr.env
Environment=PATH=$CONDA_DIR/envs/main/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$REPO
Environment=APP_ROOT=$REPO
Environment=TOOLS_DIR=$REPO/scripts/zaratan
Environment=LP3D_ROOT=$LP3D_ROOT
Environment=LP3D_CHECKPOINTS=$LP3D_CHECKPOINTS
Environment=LP3D_PY=$CONDA_DIR/envs/lp3d/bin/python
Environment=MAIN_PY=$CONDA_DIR/envs/main/bin/python
Environment=HF_HOME=$HF_HOME
Environment=SCENES_DIR=$SCENES_DIR
Environment=WORK_DIR=/mnt/work/textworld_work
Environment=PANO_DEVICE=cuda
Environment=PANO_DTYPE=float16
Environment=SCENES_BACKEND=s3
Environment=PIPELINE_CONFIG=$REPO/configs/default.yaml
ExecStart=$REPO/deploy/nebius/on_vm/run.sh
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable textworld-vr.service

echo
echo "== install.sh done"
echo "   start:   sudo systemctl start textworld-vr"
echo "   logs:    journalctl -u textworld-vr -f"
echo "   restart: sudo systemctl restart textworld-vr   (after rsync of new code)"
