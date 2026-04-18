#!/bin/bash
#SBATCH --job-name=persist-tw-vr
#SBATCH --partition=gpu-h100
#SBATCH --account=mqzhu-prj-aac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/scratch/phase4/textworld-vr/logs/%x.%j.%N.out
#SBATCH --error=/home/%u/scratch/phase4/textworld-vr/logs/%x.%j.%N.err
# ─────────────────────────────────────────────────────────────────────────
# Persistent 24h TextWorld VR worker on H100 × 4.
# Intended to be submitted twice — one job per node — so the two nodes
# can share state via ~/scratch (BeeGFS) + optional sshfs cross-mount.
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODE="${MODE:-idle}"     # 'idle' = just bootstrap + sleep; 'pipeline' = run run_pipeline.sh loop
CODE_ROOT="$HOME/scratch/phase4/textworld-vr/code"

echo "== $(date) == starting on $(hostname) (JOB_ID=$SLURM_JOB_ID, MODE=$MODE)"
echo "== CUDA modules: $(module list 2>&1 | grep cuda)"
echo "== GPUs:"
nvidia-smi -L

mkdir -p "$HOME/scratch/phase4/textworld-vr/logs"

# Bootstrap node-local workspace
bash "$CODE_ROOT/scripts/zaratan/bootstrap_node.sh"

# Periodic outputs sync: /tmp → ~/scratch, runs in background
nohup bash "$CODE_ROOT/scripts/zaratan/sync_out.sh" > \
    "$HOME/scratch/phase4/textworld-vr/logs/sync-$SLURM_JOB_ID-$(hostname).log" 2>&1 &
SYNC_PID=$!
echo "== sync_out.sh daemon PID=$SYNC_PID"

# Trap to flush + cleanup
trap 'echo "== $(date) == cleanup"; kill $SYNC_PID 2>/dev/null || true; \
      bash "$CODE_ROOT/scripts/zaratan/sync_out.sh" --once 2>&1 | tail -5' \
     EXIT TERM INT

case "$MODE" in
    idle)
        echo "== idle mode: keeping node warm for interactive ssh"
        # Sleep for 24h, or until SLURM kills us
        while sleep 900; do
            date
        done
        ;;
    pipeline)
        echo "== pipeline mode: dispatching scenes from scenes.txt"
        bash "$CODE_ROOT/scripts/zaratan/run_pipeline_loop.sh"
        ;;
    *)
        echo "== unknown MODE=$MODE; sleeping"; sleep 86400
        ;;
esac
