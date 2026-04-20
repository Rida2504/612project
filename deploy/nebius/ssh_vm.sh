#!/bin/bash
# ssh_vm.sh — quick SSH into the VM. Forwards the optional command.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "Missing .env"; exit 1; }
[ -f .vm_state.env ] || { echo "Missing .vm_state.env (run provision_vm.sh first)"; exit 1; }
set -a; source .env; source .vm_state.env; set +a
: "${PUBLIC_IP:?}"
: "${SSH_KEY_PATH:?}"
exec ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=accept-new "ubuntu@$PUBLIC_IP" "$@"
