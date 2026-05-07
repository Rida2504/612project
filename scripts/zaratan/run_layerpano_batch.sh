#!/bin/bash
# run_layerpano_batch.sh — parallel 3DGS training of layered scenes across GPUs.
#
# Each scene: run_layerpano.py on the scene's layering dir → produces
# scene/gsplat_layer{0..N}.ply into the node-local output dir.
#
# Usage: ./run_layerpano_batch.sh <scene_list.txt> [<gpu_count>]
set -u

LIST="${1:?need scene list file}"
N_GPU="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
N_GPU="${N_GPU:-1}"

LP3D_ROOT="${LP3D_ROOT:-/home/yog/scratch/phase4/textworld-vr/shared/LayerPano3D}"
OUT_ROOT="${OUT_ROOT:-/scratch/local/19035389/layered_splats}"
LOG_ROOT="${LOG_ROOT:-/home/yog/scratch/phase4/textworld-vr/logs/layerpano_batch}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

source "$HOME/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate lp3d
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libglog.so"
export HF_HOME=/home/yog/scratch/phase4/textworld-vr/shared/hf_cache
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

train_scene() {
    local scene_dir="$1"
    local gpu="$2"
    local name=$(basename "$scene_dir")
    local out_dir="$OUT_ROOT/$name"
    local log="$LOG_ROOT/$name.log"
    mkdir -p "$out_dir"
    echo "[gpu=$gpu] START $name" | tee -a "$log"
    cd "$LP3D_ROOT" || return 99
    CUDA_VISIBLE_DEVICES="$gpu" python run_layerpano.py \
        --input_dir "$scene_dir/layering" \
        --save_dir "$out_dir" \
        --outlier_thresh 4 >> "$log" 2>&1 \
        && echo "[gpu=$gpu] DONE $name" | tee -a "$log" \
        || { echo "[gpu=$gpu] FAIL $name" | tee -a "$log"; return 1; }
}

export -f train_scene
export LP3D_ROOT OUT_ROOT LOG_ROOT

mapfile -t SCENES < "$LIST"
echo "[lp-batch] scenes=${#SCENES[@]} gpus=$N_GPU"
for gpu in $(seq 0 $((N_GPU-1))); do
    (
        for i in "${!SCENES[@]}"; do
            if (( i % N_GPU == gpu )); then
                train_scene "${SCENES[$i]}" "$gpu"
            fi
        done
    ) &
done
wait
echo "[lp-batch] ALL DONE  $(date -Is)"
