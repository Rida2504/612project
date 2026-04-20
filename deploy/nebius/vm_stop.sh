#!/bin/bash
# vm_stop.sh — stop the VM (boot disk persists, no GPU charges).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .vm_state.env ] || { echo "Missing .vm_state.env"; exit 1; }
set -a; source .vm_state.env; set +a
: "${VM_ID:?}"
nebius compute instance stop --id "$VM_ID"
echo "== stopped"
