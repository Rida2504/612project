#!/bin/bash
# sync_out.sh — background rsync daemon
# Pushes /tmp/$USER/textworld-vr/outputs/ → ~/scratch/.../outputs/ every N seconds.
# Also pushes HF cache back to shared storage so other nodes can reuse downloads.
#
# Usage:
#   sync_out.sh           # daemon mode (infinite loop, 60s interval)
#   sync_out.sh --once    # single pass and exit (for cleanup trap)
#   sync_out.sh --interval 30   # custom interval in seconds

set -euo pipefail

INTERVAL="${INTERVAL:-60}"
ONCE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --once) ONCE=true; shift ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

SCRATCH_ROOT="$HOME/scratch/phase4/textworld-vr"
TMP_ROOT="/tmp/$USER/textworld-vr"

do_sync() {
    # 1. Outputs: push /tmp → ~/scratch. We only PUSH (no pull) so local work
    #    isn't clobbered; cross-node consistency via BeeGFS visibility.
    if [ -d "$TMP_ROOT/outputs" ]; then
        rsync -a --no-perms --no-owner --no-group \
              --exclude='tmp_*' --exclude='*.tmp' \
              "$TMP_ROOT/outputs/" "$SCRATCH_ROOT/outputs/" 2>&1 | tail -5 || true
    fi

    # 2. HF cache: push hot downloads back to shared storage for other nodes.
    #    Use --size-only + --ignore-existing to avoid clobbering partial writes.
    if [ -d "$TMP_ROOT/hf_cache" ]; then
        rsync -a --size-only --ignore-existing \
              "$TMP_ROOT/hf_cache/" "$SCRATCH_ROOT/shared/hf_cache/" 2>&1 | tail -3 || true
    fi
}

echo "[sync_out] $(date -Is) starting on $(hostname); interval=${INTERVAL}s once=${ONCE}"

if $ONCE; then
    do_sync
    echo "[sync_out] $(date -Is) one-shot done"
    exit 0
fi

while true; do
    do_sync
    echo "[sync_out] $(date -Is) pass complete"
    sleep "$INTERVAL"
done
