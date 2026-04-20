# TextWorld VR — Nebius VM deploy

Single GPU VM, no docker, no Ray. Conda envs + systemd unit + nginx. Pay only
for on-minutes (~$2-3/hr for RTX PRO 6000 Blackwell). Boot disk persists across
stop/start so the 80 GB model cache stays warm.

## One-time setup (laptop)

```bash
cd textworld-vr/deploy/nebius
cp .env.example .env       # already populated for the BrainGnosis project
                           # check IDs/region match your Nebius profile
./provision_vm.sh          # creates 500 GB SSD disk + GPU VM, prints public IP
                           # writes .vm_state.env (VM_ID, DISK_ID, PUBLIC_IP)
```

Wait ~3-5 min for cloud-init to install Docker (still useful for tooling) +
NVIDIA toolkit. The script polls until SSH works and prints the IP.

```bash
./sync_to_vm.sh            # rsync source tree -> /mnt/src/textworld-vr on VM
```

Then push the app secrets file (HF token + S3 creds) to the VM:

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token)
ssh ubuntu@<vm-ip> "cat > ~/textworld-vr.env" <<EOF
HF_TOKEN=$HF_TOKEN
S3_BUCKET=textworld-vr-scenes
S3_ENDPOINT_URL=https://storage.us-central1.nebius.cloud
S3_PREFIX=scenes/
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-central1
EOF
ssh ubuntu@<vm-ip> 'chmod 600 ~/textworld-vr.env'
```

SSH in and run the native installer (one-time, ~20-30 min):

```bash
./ssh_vm.sh
cd /mnt/src/textworld-vr/deploy/nebius/on_vm
./install.sh               # Miniconda + 2 conda envs + LayerPano3D +
                           # 360monodepth C++ ext + diff-gaussian-rasterization +
                           # nginx config + systemd unit
sudo systemctl start textworld-vr
journalctl -u textworld-vr -f
```

First start does the model staging (~80 GB HF download, ~10 min on Nebius
internal network). It only runs once — the cache lives on the boot disk.

## Daily loop

```bash
# Resume after stopping
./vm_start.sh

# Iterate on code (~2-second restart, no rebuild)
./sync_to_vm.sh
./ssh_vm.sh 'sudo systemctl restart textworld-vr && journalctl -u textworld-vr -f'

# Smoke test
PUBLIC_IP=$(grep PUBLIC_IP .vm_state.env | cut -d'"' -f2)
curl -X POST "http://$PUBLIC_IP:8000/generate" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a cozy japanese coffee shop","seed":42}'
# poll http://$PUBLIC_IP:8000/status/<job_id> until state == "done"
# scene_url is a presigned Nebius S3 URL; open in WebXR viewer at
#   http://$PUBLIC_IP:8080/viewer/

# Done for the day
./vm_stop.sh               # GPU charges stop; ~$5/month for the 500 GB disk
```

## File layout on the VM

```
/opt/conda/                          Miniconda
/opt/conda/envs/main                 Python 3.12, torch+cu121, FastAPI, gsplat, diff-gaussian-rasterization
/opt/conda/envs/lp3d                 Python 3.9,  torch+cu118, LP3D deps
/opt/LayerPano3D/                    LP3D source + 360monodepth C++ ext
/mnt/models/hf_cache/                HF cache (persistent on boot disk)
/mnt/scenes/                         local scene fallback (S3 path bypasses)
/mnt/work/                           scratch (auto-cleaned per job)
/mnt/src/textworld-vr/               code (rsynced from laptop)
/etc/systemd/system/textworld-vr.service   systemd unit
/etc/nginx/sites-enabled/textworld-vr      nginx config
/var/www/textworld-vr/viewer/        static WebXR viewer assets
~/textworld-vr.env                   HF token + S3 creds (chmod 600)
```

## Helpers

| Script | Runs where | Purpose |
|---|---|---|
| `provision_vm.sh` | laptop | create disk + VM (idempotent) |
| `sync_to_vm.sh` | laptop | rsync source tree |
| `ssh_vm.sh` | laptop | quick SSH wrapper |
| `vm_start.sh` / `vm_stop.sh` | laptop | lifecycle |
| `on_vm/install.sh` | VM | native install (one-time) |

## Why no docker

The original Dockerfile existed because Nebius Serverless AI Endpoints require
a container image. On a dedicated VM, docker only adds ~5-15 min per code
iteration (image rebuild) for zero benefit — every step in the Dockerfile is
already a shell command, and conda gives us hermetic Python envs without the
container layer. See [the parent transcript discussion](../../README.md) for
the full reasoning.

The Dockerfile, `entrypoint.sh`, and `on_vm/build_and_run.sh` are kept as a
fallback for the AI Endpoint deploy path (`./deploy.sh`), which still works
once you build the image.
