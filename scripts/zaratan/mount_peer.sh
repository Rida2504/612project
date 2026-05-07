#!/bin/bash
# mount_peer.sh <peer-hostname>
# sshfs the peer node's /tmp/$USER/textworld-vr onto the local node at
# /tmp/$USER/textworld-vr/peer-<host>/. No root needed.
#
# Usage from gpu-a6-8:
#   bash mount_peer.sh gpu-a6-9
# Usage from gpu-a6-9:
#   bash mount_peer.sh gpu-a6-8
#
# Assumes the peer node is already running and has bootstrap_node.sh applied.
# Needs passwordless ssh between compute nodes (standard on Zaratan for
# same-user jobs).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <peer-hostname>    (e.g. gpu-a6-9)"
    exit 2
fi

PEER="$1"
PEER_SHORT="${PEER%%.*}"
LOCAL_MOUNT="/tmp/$USER/textworld-vr/peer-$PEER_SHORT"
REMOTE_PATH="/tmp/$USER/textworld-vr"

if ! command -v sshfs >/dev/null; then
    echo "sshfs not installed on $(hostname). Trying module..."
    module load sshfs 2>/dev/null || true
    if ! command -v sshfs >/dev/null; then
        echo "ERROR: sshfs missing. Alternatives:"
        echo "  - ask hpc@umd to install sshfs, OR"
        echo "  - use rsync-based sync only (sync_out.sh) instead of a live mount"
        exit 1
    fi
fi

mkdir -p "$LOCAL_MOUNT"

# Already mounted?
if mountpoint -q "$LOCAL_MOUNT"; then
    echo "already mounted: $LOCAL_MOUNT"
    ls -la "$LOCAL_MOUNT" | head -3
    exit 0
fi

echo "mounting $PEER:$REMOTE_PATH → $LOCAL_MOUNT"
sshfs "$PEER:$REMOTE_PATH" "$LOCAL_MOUNT" \
    -o reconnect \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o cache_timeout=10 \
    -o entry_timeout=10 \
    -o attr_timeout=10

echo "mounted."
ls "$LOCAL_MOUNT" | head -5
