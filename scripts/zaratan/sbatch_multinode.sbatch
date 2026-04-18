#!/bin/bash
#SBATCH --job-name=persist-tw-vr-2n
#SBATCH --partition=gpu-h100
#SBATCH --account=mqzhu-prj-aac
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/scratch/phase4/textworld-vr/logs/%x.%j.out
#SBATCH --error=/home/%u/scratch/phase4/textworld-vr/logs/%x.%j.err
# ─────────────────────────────────────────────────────────────────────────
# One 24h job spanning 2 H100 nodes. --exclusive + --nodes=2 enables
# Zaratan's native /scratch/local/$SLURM_JOB_ID striped scratch
# (combined /tmp of both nodes, ~24 TB).
#
# All cost charged to mqzhu-prj-aac.
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

CODE_ROOT="$HOME/scratch/phase4/textworld-vr/code"
STRIPED="/scratch/local/$SLURM_JOB_ID"

echo "== $(date) == starting job $SLURM_JOB_ID"
echo "== nodes: $SLURM_NODELIST"
echo "== expanded: $(scontrol show hostnames $SLURM_NODELIST | tr '\n' ' ')"
echo "== striped scratch: $STRIPED"

mkdir -p "$HOME/scratch/phase4/textworld-vr/logs"

# Bootstrap on ALL nodes in parallel. srun broadcasts the command.
echo "== bootstrapping all nodes via srun =="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --label \
     bash "$CODE_ROOT/scripts/zaratan/bootstrap_node.sh"

# Verify striped /scratch/local exists and is shared
echo "== verifying striped scratch on all nodes =="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --label bash -c \
     "hostname; ls -ld $STRIPED 2>&1 | head -1; df -h $STRIPED 2>&1 | tail -1"

# Seed striped scratch with code (each node reads the same file via striping)
echo "== seeding striped scratch with code =="
mkdir -p "$STRIPED/code" "$STRIPED/outputs" "$STRIPED/hf_cache"
rsync -a --exclude='outputs/' --exclude='.git/' --exclude='__pycache__/' \
      "$CODE_ROOT/" "$STRIPED/code/"

# Start the background sync daemon here (the sbatch script runs on the batch host
# = first allocated node). This process lives for the lifetime of the job.
echo "== starting sync daemon =="
bash "$CODE_ROOT/scripts/zaratan/sync_out.sh" \
     > "$HOME/scratch/phase4/textworld-vr/logs/sync-$SLURM_JOB_ID.log" 2>&1 &
SYNC_PID=$!
echo "== sync daemon PID=$SYNC_PID"

# Idle loop: keep the job alive for interactive ssh-in. Each node can be
# reached via scontrol show hostnames; user ssh's in to run experiments.
echo "== entering idle loop (24h) — ssh into the allocated nodes to work =="
scontrol show hostnames $SLURM_NODELIST > "$HOME/scratch/phase4/textworld-vr/logs/nodes-$SLURM_JOB_ID.txt"

trap 'echo "== $(date) == final sync"; \
      bash "$CODE_ROOT/scripts/zaratan/sync_out.sh" --once 2>&1 | tail -5' \
     EXIT TERM INT

while sleep 900; do
    date
    # Keep last-modified timestamp updated so user can sanity-check the job is alive
    touch "$HOME/scratch/phase4/textworld-vr/logs/heartbeat-$SLURM_JOB_ID"
done
