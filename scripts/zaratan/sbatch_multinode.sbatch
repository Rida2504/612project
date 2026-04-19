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
#SBATCH --constraint=beeond
#SBATCH --time=23:00:00
#SBATCH --output=/home/%u/scratch/phase4/textworld-vr/logs/%x.%j.out
#SBATCH --error=/home/%u/scratch/phase4/textworld-vr/logs/%x.%j.err
# ─────────────────────────────────────────────────────────────────────────
# 23h persistent job on 2× H100 nodes with BeeOND.
#
# --constraint=beeond + --exclusive + --nodes=2 tells Zaratan to stripe the
# node-local /tmp of every allocated node into a single shared filesystem
# at /scratch/local/$SLURM_JOB_ID, mounted read-write from all nodes.
#
# See https://hpcc.umd.edu/kb/storage/#beeond-multi-node-high-performance-temporary-storage
# ─────────────────────────────────────────────────────────────────────────

set -uo pipefail   # survive partial failures

CODE_ROOT="$HOME/scratch/phase4/textworld-vr/code"
SCRATCH_ROOT="$HOME/scratch/phase4/textworld-vr"
STRIPED="/scratch/local/$SLURM_JOB_ID"
LOG_DIR="$SCRATCH_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "== $(date) == starting job $SLURM_JOB_ID"
echo "== nodes: $SLURM_NODELIST"
echo "== expanded: $(scontrol show hostnames $SLURM_NODELIST | tr '\n' ' ')"
echo "== BeeOND expected at: $STRIPED"

# Probe BeeOND from every node. If it's not there, fail loudly rather than
# silently falling back — we chose the proper solution, we want it to work.
echo "== verifying BeeOND is mounted on all nodes =="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --label bash -c "
    hostname
    if [ ! -d '$STRIPED' ]; then
        echo 'ERROR: BeeOND not mounted at $STRIPED on \$(hostname)'
        exit 1
    fi
    df -h '$STRIPED' | tail -1
    stat -c 'mounted-type=%T' '$STRIPED'
" 2>&1 | tee "$LOG_DIR/beeond-probe-$SLURM_JOB_ID.log"

if [ ! -d "$STRIPED" ]; then
    echo "FATAL: BeeOND not available. Check partition/QOS supports -C beeond."
    exit 1
fi

# Seed BeeOND with the workspace. Rank-0 does it once; every other node sees
# it instantly because /scratch/local/$JOBID is a single shared filesystem.
echo "== seeding BeeOND workspace from ~/scratch =="
mkdir -p "$STRIPED"/{workdir,hf_cache,logs}
rsync -a --delete \
    --exclude='outputs/' --exclude='.git/' --exclude='__pycache__/' \
    "$CODE_ROOT/" "$STRIPED/workdir/code/"
mkdir -p "$STRIPED/workdir/outputs"
# Mirror the durable outputs dir so jobs can append without losing prior work
rsync -a "$SCRATCH_ROOT/outputs/" "$STRIPED/workdir/outputs/" || true

echo "== BeeOND workspace layout =="
ls -la "$STRIPED"
du -sh "$STRIPED"/* 2>/dev/null

# Bootstrap env on every node (sets HF_HOME, venv path, etc.)
echo "== bootstrapping env on all nodes via srun =="
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --label \
    env TW_STRIPED="$STRIPED" bash "$CODE_ROOT/scripts/zaratan/bootstrap_node.sh"

# Start sync daemon on batch host (rank-0). Since BeeOND is shared across
# nodes, ONE daemon is enough — it sees all writes from all nodes.
echo "== starting single sync daemon (BeeOND is shared, no per-node needed) =="
tmux kill-session -t sync 2>/dev/null || true
tmux new-session -d -s sync "
    env TW_STRIPED='$STRIPED' bash $CODE_ROOT/scripts/zaratan/sync_out.sh 2>&1 \
      | tee $LOG_DIR/sync-$SLURM_JOB_ID.log
"
echo "== sync daemon tmux session started: $(tmux ls 2>&1 | head -3)"

# Record allocated node list for SSH aliases
scontrol show hostnames $SLURM_NODELIST > "$LOG_DIR/nodes-$SLURM_JOB_ID.txt"
echo "== allocated nodes =="
cat "$LOG_DIR/nodes-$SLURM_JOB_ID.txt"

trap '
    echo "== $(date) == cleanup: final sync from BeeOND to ~/scratch"
    tmux kill-session -t sync 2>/dev/null || true
    env TW_STRIPED="'"$STRIPED"'" bash "$CODE_ROOT/scripts/zaratan/sync_out.sh" --once 2>&1 | tail -5 || true
    echo "== BeeOND will be auto-torn-down by scheduler =="
' EXIT TERM INT

echo "== entering idle loop (23h). ssh into allocated nodes to work. =="
echo "== workspace: $STRIPED/workdir/"
touch "$LOG_DIR/heartbeat-$SLURM_JOB_ID"
while sleep 600; do
    date >> "$LOG_DIR/heartbeat-$SLURM_JOB_ID"
done
