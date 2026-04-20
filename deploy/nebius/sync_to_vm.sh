#!/bin/bash
# sync_to_vm.sh — rsync the source tree to /mnt/src on the VM.
#
# Run this every time you change code locally and want to rebuild the image
# on the VM. Cheap (only ships diffs).

set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] || { echo "Missing .env"; exit 1; }
[ -f .vm_state.env ] || { echo "Missing .vm_state.env (run provision_vm.sh first)"; exit 1; }
set -a; source .env; source .vm_state.env; set +a

: "${PUBLIC_IP:?}"
: "${SSH_KEY_PATH:?}"

# Repo root is two levels up from this script (deploy/nebius/sync_to_vm.sh).
REPO_ROOT="$(cd ../../.. && pwd)"
echo "== syncing $REPO_ROOT/textworld-vr -> ubuntu@$PUBLIC_IP:/mnt/src/textworld-vr"

# Excludes: outputs, __pycache__, .git (we send code, not history),
# already-staged model caches, scene blobs.
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'outputs/' \
    --exclude 'models/' \
    --exclude 'scenes/' \
    --exclude '.DS_Store' \
    --exclude 'deploy/nebius/.env' \
    --exclude 'deploy/nebius/.vm_state.env' \
    --exclude 'deploy/nebius/.secrets/' \
    -e "ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=accept-new" \
    "$REPO_ROOT/textworld-vr/" "ubuntu@$PUBLIC_IP:/mnt/src/textworld-vr/"

echo "== done"
echo "   First time on VM:  cd /mnt/src/textworld-vr/deploy/nebius/on_vm && ./install.sh"
echo "   After code change:  ssh ... 'sudo systemctl restart textworld-vr && journalctl -u textworld-vr -f'"
