#!/bin/bash
# run_batch_multigpu.sh — dispatch run_pipeline jobs across multiple GPUs on a node.
#
# Each local GPU is a dedicated worker that runs batch_worker.py ONCE and
# streams its tasks via stdin. This keeps CUDA context + gsplat JIT cache
# warm across the tasks that belong to this worker, instead of paying the
# ~60s context-init cost per task.
#
# Tasks are distributed across a GLOBAL worker pool via awk line-number mod:
# worker `global_idx` processes tasks where `NR%TOTAL_WORKERS==global_idx`.
# With two nodes, pass --global-offset 0 on node A and --global-offset 4 on
# node B (and --total-workers 8) so they cover disjoint subsets.
#
# Each worker also gets its own TORCH_EXTENSIONS_DIR so multiple workers on
# the same node can't race on the torch_extensions JIT build dir — a real
# crash we observed (ImportError: gsplat_cuda.so: cannot open shared object
# file) when 8 workers shared $HOME/.cache/torch_extensions.
#
# Usage:
#   run_batch_multigpu.sh --gpus 0,1,2,3 --global-offset 0 --total-workers 8 \
#                        --tasks /path/to/tasks.tsv --out-dir /path/to/outputs
#
# Example (2-node, 8 GPUs total):
#   Node A (gpu-a6-8): --gpus 0,1,2,3 --global-offset 0 --total-workers 8
#   Node B (gpu-a6-9): --gpus 0,1,2,3 --global-offset 4 --total-workers 8

set -uo pipefail

GPUS="0,1,2,3"
GLOBAL_OFFSET=0
TOTAL_WORKERS=4
TASKS=""
OUTDIR=""
ITERS=1500
MAX_GAUSSIANS=300000
SUFFIX=""

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --global-offset) GLOBAL_OFFSET="$2"; shift 2 ;;
        --total-workers) TOTAL_WORKERS="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --out-dir) OUTDIR="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --max-gaussians) MAX_GAUSSIANS="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$TASKS" ] || [ -z "$OUTDIR" ]; then
    echo "ERROR: --tasks and --out-dir are required" >&2
    exit 2
fi

mkdir -p "$OUTDIR/logs" "$OUTDIR/torch_ext"
IFS=',' read -ra GPU_ARR <<< "$GPUS"

echo "[$(date -Is)] host=$(hostname) gpus=$GPUS offset=$GLOBAL_OFFSET workers=$TOTAL_WORKERS tasks=$TASKS"
echo "[$(date -Is)] tasks.tsv line count: $(wc -l < "$TASKS")"

for local_i in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$local_i]}"
    global_idx=$((GLOBAL_OFFSET + local_i))
    remainder=$((global_idx % TOTAL_WORKERS))

    log="$OUTDIR/logs/worker_global${global_idx}_gpu${GPU}${SUFFIX}.log"
    taskfile="$OUTDIR/logs/worker_global${global_idx}_gpu${GPU}${SUFFIX}.tasks.tsv"
    worker_ext="$OUTDIR/torch_ext/global${global_idx}"
    mkdir -p "$worker_ext"

    # Extract this worker's task subset
    awk -v k=$TOTAL_WORKERS -v r=$remainder 'NR%k==r%k' "$TASKS" > "$taskfile"
    n_tasks=$(wc -l < "$taskfile")
    echo "[gpu$GPU|global$global_idx|$(hostname)] dispatching $n_tasks tasks → $log"

    (
        exec > "$log" 2>&1
        export CUDA_VISIBLE_DEVICES=$GPU
        export TORCH_EXTENSIONS_DIR="$worker_ext"
        export TORCH_CUDA_ARCH_LIST="9.0"
        export BATCH_TAG="gpu${GPU}|global${global_idx}"
        echo "[$(date -Is)] worker START global=$global_idx gpu=$GPU ext_dir=$worker_ext tasks=$n_tasks"
        cat "$taskfile" | python scripts/zaratan/batch_worker.py \
            --config configs/default.yaml \
            --output-dir "$OUTDIR" \
            --device cuda \
            --iters "$ITERS" \
            --max-gaussians "$MAX_GAUSSIANS" \
            --tag "gpu${GPU}|global${global_idx}"
        ec=$?
        echo "[$(date -Is)] worker END global=$global_idx exit=$ec"
    ) &
done

wait
echo "[$(date -Is)] all GPU workers done on $(hostname)"
