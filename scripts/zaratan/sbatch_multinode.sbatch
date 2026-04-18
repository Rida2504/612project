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
#SBATCH --tmp=4000G
#SBATCH --time=23:00:00
#SBATCH --output=/home/%u/scratch/phase4/textworld-vr/logs/%x.%j.out
#SBATCH --error=/home/%u/scratch/phase4/textworld-vr/logs/%x.%j.err
# ─────────────────────────────────────────────────────────────────────────
# 23h persistent job on 2× H100 nodes. The --tmp=4000G + --exclusive +
# --nodes=2 combination triggers Zaratan's native striped scratch at
# /scratch/local/$SLURM_JOB_ID (combined /tmp of both nodes).
#
# If striping isn't created (older kernel, bad node), the script falls
# back gracefully to per-node /tmp/$USER with a BeeGFS sync daemon.
# ─────────────────────────────────────────────────────────────────────────

set -uo pipefail   # note: no -e, we want to survive partial failures

CODE_ROOT="$HOME/scratch/phase4/textworld-vr/code"
STRIPED="/scratch/local/$SLURM_JOB_ID"
LOG_DIR="$HOME/scratch/phase4/textworld-vr/logs"
mkdir -p "$LOG_DIR"

echo "== $(date) == starting job $SLURM_JOB_ID"
echo "== nodes: $SLURM_NODELIST"
echo "== expanded: $(scontrol show hostnames $SLURM_NODELIST | tr '\n' ' ')"

# Bootstrap each node's local /tmp workspace. Runs in parallel via srun.
echo "== bootstrapping all nodes via srun =="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --label \
     bash "$CODE_ROOT/scripts/zaratan/bootstrap_node.sh" || \
     echo "!! bootstrap_node.sh had errors, continuing"

# Probe for striped /scratch/local on each node
echo "== probing striped scratch: $STRIPED =="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --label bash -c \
    "hostname; if [ -d '$STRIPED' ]; then echo 'STRIPED_OK'; df -h '$STRIPED' | tail -1; \
     else echo 'NO_STRIPED'; fi" 2>&1 | tee "$LOG_DIR/striped-probe-$SLURM_JOB_ID.log"

USE_STRIPED=false
if srun --nodes=1 --ntasks=1 test -d "$STRIPED"; then
    USE_STRIPED=true
    echo "== striped scratch available, using $STRIPED as workspace =="
    mkdir -p "$STRIPED/code" "$STRIPED/outputs" "$STRIPED/hf_cache" 2>/dev/null || true
    rsync -a --exclude='outputs/' --exclude='.git/' --exclude='__pycache__/' \
          "$CODE_ROOT/" "$STRIPED/code/" || echo "!! rsync to striped failed"
    echo "STRIPED_WORKSPACE=$STRIPED" > "$LOG_DIR/workspace-$SLURM_JOB_ID.env"
else
    echo "== falling back to per-node /tmp + BeeGFS sync (Path 1) =="
    echo "WORKSPACE_PER_NODE_TMP=/tmp/$USER/textworld-vr" > "$LOG_DIR/workspace-$SLURM_JOB_ID.env"
fi

# Start the background sync daemon (rsync /tmp → ~/scratch every 60s)
echo "== starting sync daemon on batch host ($(hostname)) =="
bash "$CODE_ROOT/scripts/zaratan/sync_out.sh" \
    > "$LOG_DIR/sync-$SLURM_JOB_ID.log" 2>&1 &
SYNC_PID=$!
echo "== sync daemon PID=$SYNC_PID"

# Record allocated node list in a predictable file for SSH aliases
scontrol show hostnames $SLURM_NODELIST > "$LOG_DIR/nodes-$SLURM_JOB_ID.txt"
echo "== nodes file: $LOG_DIR/nodes-$SLURM_JOB_ID.txt"
cat "$LOG_DIR/nodes-$SLURM_JOB_ID.txt"

trap '
    echo "== $(date) == cleanup: final sync + daemon kill"
    kill $SYNC_PID 2>/dev/null || true
    bash "$CODE_ROOT/scripts/zaratan/sync_out.sh" --once 2>&1 | tail -5 || true
' EXIT TERM INT

# Idle loop: 23h heartbeat. User ssh-es into either allocated node to work.
echo "== entering idle loop (23h) =="
touch "$LOG_DIR/heartbeat-$SLURM_JOB_ID"
while sleep 600; do
    date >> "$LOG_DIR/heartbeat-$SLURM_JOB_ID"
done
