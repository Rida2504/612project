#!/bin/bash
# umount_peer.sh — tear down peer sshfs mounts.
set -euo pipefail
for m in /tmp/$USER/textworld-vr/peer-*/; do
    if mountpoint -q "$m" 2>/dev/null; then
        echo "unmounting $m"
        fusermount -u "$m" || fusermount3 -u "$m" || true
    fi
done
