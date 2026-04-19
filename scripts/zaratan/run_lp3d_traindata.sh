#!/bin/bash
# run_lp3d_traindata.sh — gen_traindata (Infusion) for a batch of already-layered scenes.
#
# Usage: ./run_lp3d_traindata.sh <scene_list.txt> [<gpu_count>]
# <scene_list.txt> lists absolute paths to scene dirs that ALREADY have
# layering/layer{0,1,2}/ populated by run_lp3d_batch.sh.
set -u

LIST="${1:?need scene list file}"
N_GPU="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
N_GPU="${N_GPU:-1}"

LP3D_ROOT="${LP3D_ROOT:-/home/yog/scratch/phase4/textworld-vr/shared/LayerPano3D}"
LOG_ROOT="${LOG_ROOT:-/home/yog/scratch/phase4/textworld-vr/logs/lp3d_traindata}"
mkdir -p "$LOG_ROOT"

source "$HOME/scratch/miniconda3/etc/profile.d/conda.sh"
conda activate lp3d
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libglog.so"
export HF_HOME=/home/yog/scratch/phase4/textworld-vr/shared/hf_cache
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

process() {
    local scene_dir="$1"
    local gpu="$2"
    local name=$(basename "$scene_dir")
    local log="$LOG_ROOT/$name.log"
    echo "[gpu=$gpu] START traindata $name" | tee -a "$log"
    cd "$LP3D_ROOT" || return 99
    CUDA_VISIBLE_DEVICES="$gpu" python gen_traindata.py \
        --layerpano_dir "$scene_dir/layering" \
        --save_dir "$scene_dir/layering" \
        --root "$scene_dir/layering" >> "$log" 2>&1 \
        && echo "[gpu=$gpu] traindata OK $name" | tee -a "$log" \
        || { echo "[gpu=$gpu] FAIL traindata $name" | tee -a "$log"; return 1; }
    echo "[gpu=$gpu] DONE $name" | tee -a "$log"
}

export -f process
export LP3D_ROOT LOG_ROOT

mapfile -t SCENES < "$LIST"
echo "[td-batch] scenes=${#SCENES[@]} gpus=$N_GPU"
for gpu in $(seq 0 $((N_GPU-1))); do
    (
        for i in "${!SCENES[@]}"; do
            if (( i % N_GPU == gpu )); then
                process "${SCENES[$i]}" "$gpu"
            fi
        done
    ) &
done
wait
echo "[td-batch] ALL DONE  $(date -Is)"
