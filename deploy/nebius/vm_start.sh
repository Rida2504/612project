#!/bin/bash
# vm_start.sh — start the VM (you pay only for on-minutes).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "Missing .env"; exit 1; }
[ -f .vm_state.env ] || { echo "Missing .vm_state.env"; exit 1; }
set -a; source .env; source .vm_state.env; set +a
: "${VM_ID:?}"
nebius compute instance start --id "$VM_ID"
echo "== started; waiting for SSH"
for i in $(seq 1 60); do
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -i "$SSH_KEY_PATH" "ubuntu@$PUBLIC_IP" 'true' 2>/dev/null; then
        echo "   ready: ssh ubuntu@$PUBLIC_IP"
        exit 0
    fi
    sleep 5
done
echo "Warning: VM started but SSH not reachable after 5 min"
