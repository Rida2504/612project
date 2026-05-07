#!/bin/bash
# sync_out.sh — daemon that mirrors workspace outputs + HF cache from the
# shared TW_TMP (BeeOND) back to durable ~/scratch. 60s interval.
#
# Under BeeOND, TW_TMP is /scratch/local/$SLURM_JOB_ID/workdir and only ONE
# instance of this daemon is needed per job (since it's shared, not per-node).
# Under /tmp fallback, TW_TMP is /tmp/$USER/textworld-vr and the daemon must
# run on EACH node (the sbatch wrapper handles that).

set -uo pipefail

INTERVAL="${INTERVAL:-60}"
ONCE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --once) ONCE=true; shift ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

SCRATCH_ROOT="$HOME/scratch/phase4/textworld-vr"

# Pick TW_TMP from env (set by bootstrap_node.sh) or derive
if [ -n "${TW_STRIPED:-}" ] && [ -d "$TW_STRIPED" ]; then
    TW_TMP="$TW_STRIPED/workdir"
    TW_HF_CACHE="$TW_STRIPED/hf_cache"
    MODE="beeond"
elif [ -n "${TW_TMP:-}" ]; then
    TW_HF_CACHE="${TW_TMP}/hf_cache"
    MODE="tmp-env"
else
    TW_TMP="/tmp/$USER/textworld-vr"
    TW_HF_CACHE="$TW_TMP/hf_cache"
    MODE="tmp-fallback"
fi

echo "[sync_out] $(date -Is) starting on $(hostname); mode=$MODE tw_tmp=$TW_TMP interval=${INTERVAL}s once=$ONCE"

do_sync() {
    # 1. Push outputs to durable scratch
    if [ -d "$TW_TMP/outputs" ]; then
        rsync -a --no-perms --no-owner --no-group \
              --exclude='tmp_*' --exclude='*.tmp' \
              "$TW_TMP/outputs/" "$SCRATCH_ROOT/outputs/" 2>&1 | tail -5 || true
    fi

    # 2. Push HF cache additions to the shared durable cache (for next job)
    if [ -d "$TW_HF_CACHE" ]; then
        rsync -a --size-only --ignore-existing \
              "$TW_HF_CACHE/" "$SCRATCH_ROOT/shared/hf_cache/" 2>&1 | tail -3 || true
    fi
}

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
