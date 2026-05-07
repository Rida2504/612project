# TextWorld VR — Deployment

Three paths, pick the one that matches your environment:

| Path | Best for | Scale-to-zero | Effort |
|---|---|---|---|
| **A. Local docker-compose** | demo / dev on a GPU box | n/a | 5 min |
| **B. Nebius Serverless AI endpoint** | single-instance managed, cheap idle | via stop/start watchdog (~15m) | 20 min |
| **C. Nebius MK8s + Knative** | real traffic, true autoscale 0→N | yes, per-request (~3 min cold start) | ~1 h |

All three use the **same Docker image**. The deploy knob is just how you run it.

---

## 0. Prereqs (all paths)

- HuggingFace token with FLUX licenses accepted: https://huggingface.co/black-forest-labs/FLUX.1-dev, https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
- NVIDIA GPU with ≥80 GB VRAM (H100 SXM recommended; A100 80 GB works)
- Docker with the NVIDIA Container Toolkit (`nvidia-ctk`) for local runs

## 1. Build the image

```bash
cd /path/to/Project            # parent of textworld-vr/
docker build --platform=linux/amd64 -t textworld-vr:dev -f textworld-vr/deploy/Dockerfile .
```

First build pulls ~20 GB (CUDA base + two conda envs + LP3D + gsplat). ~20 min on a fast link.

---

## Path A — Local docker-compose

```bash
cd textworld-vr/deploy
export HF_TOKEN=hf_xxx
docker compose up                       # stays attached; ^C to stop
# or
docker compose up -d                    # detached
```

First start downloads ~80 GB of models into `deploy/_models/` (one-time). Subsequent starts reuse the cache.

Ports:
- `:8000` — API (Ray Serve / FastAPI)
- `:8080` — WebXR viewer + local `/splats/` file server
- `:8265` — Ray dashboard

Smoke test:

```bash
# wait for "Deployed Serve app successfully" in logs, then:
curl -X POST http://localhost:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"a cozy japanese coffee shop","seed":42}'
# → {"job_id":"abc123", "status_url":"/status/abc123"}

curl http://localhost:8000/status/abc123   # poll until state=done
open http://localhost:8080/viewer/          # browse scenes
```

Scene storage default: `./_scenes/` (local filesystem). Flip to S3 by setting `SCENES_BACKEND=s3` + the S3 env vars in `docker-compose.yml`.

---

## Path B — Nebius Serverless AI Endpoint

**Scale-to-zero mechanism:** Nebius AI endpoints don't have request-driven scale-to-zero for custom containers yet, so we use lifecycle commands. The server exposes `/idle`; a cron job (or companion CPU endpoint) polls it and calls `nebius ai endpoint stop` when idle >15 min. Cold start back up is ~3 min (models are already on the endpoint's disk across starts, we just need app boot).

For strict per-request scale-to-zero use Path C.

### B.1 One-time setup

```bash
# Install the Nebius CLI + log in
curl -sSL https://storage.ai.nebius.cloud/cli/install | bash
nebius profile create

# Create an S3 bucket for scenes
nebius storage bucket create --name textworld-vr-scenes --parent-id $PARENT_ID
nebius iam access-key create --parent-id $PARENT_ID --format json     # save the key pair

# Create (or reuse) a Container Registry
nebius cr registry create --name textworld-vr --parent-id $PARENT_ID
REGISTRY_ID=$(nebius cr registry get-by-name --name textworld-vr --parent-id $PARENT_ID --format json | jq -r .metadata.id)
```

### B.2 Deploy

```bash
export PARENT_ID=...                    # your Nebius project id
export REGISTRY_ID=...                  # from step above
export SUBNET_ID=...                    # Nebius VPC subnet id
export HF_TOKEN=hf_xxx
export S3_BUCKET=textworld-vr-scenes
export S3_ACCESS_KEY_ID=...
export S3_SECRET_ACCESS_KEY=...
# optional:
# export S3_PUBLIC_BASE_URL=https://textworld-vr-scenes.storage.eu-north1.nebius.cloud
# export IMAGE_TAG=v1

bash textworld-vr/deploy/nebius/deploy.sh
```

`deploy.sh` will:
1. `docker build` + `docker push` to `cr.<parent>.nebiuscloud.net/<registry>/textworld-vr:<tag>`
2. Store `HF_TOKEN`, the endpoint auth token, and S3 creds in Nebius MysteryBox secrets
3. `nebius ai endpoint create` with platform=`gpu-h100-sxm`, preset=`1gpu-16vcpu-200gb`, public HTTPS, `--auth token`, `--volume s3://<bucket>:/scenes:rw:default`
4. Print the invoke URL and save the auth token

### B.3 Use it

```bash
URL=https://<endpoint>.inference.eu-north1.nebius.cloud
TOKEN=<the auth token deploy.sh printed>

curl -X POST $URL/generate \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"prompt":"a sunlit modernist kitchen","seed":42}'
```

Scenes land in `s3://textworld-vr-scenes/scenes/<slug>_s42_layered.ply` and `/scenes` returns presigned URLs (or public URLs if you set `S3_PUBLIC_BASE_URL`).

### B.4 Idle watchdog

```bash
export PARENT_ID=... ENDPOINT_URL=$URL ENDPOINT_AUTH_TOKEN=$TOKEN IDLE_MINUTES=15
# Run every 5 min via cron or Nebius schedule
textworld-vr/deploy/nebius/stop_when_idle.sh
```

To bring it back up when you need to generate:
```bash
nebius ai endpoint start --name textworld-vr --parent-id $PARENT_ID
```

---

## Path C — Nebius MK8s + Knative (per-request scale-to-zero)

True scale-to-zero: Knative's activator holds incoming requests while it spins up a pod. Cold start is ~3 min (model PVC is pre-warmed, just need app boot). First request of the day pays the cold start; subsequent requests while the pod is warm are instant.

### C.1 Cluster

```bash
# Create MK8s cluster with a GPU node pool (autoscale 0→N)
nebius mk8s cluster create --name textworld-vr --parent-id $PARENT_ID \
    --etcd-cluster-size 3 --k8s-version 1.31
nebius mk8s node-group create --name gpu-h100 --cluster-id $CLUSTER_ID \
    --platform gpu-h100-sxm --preset 1gpu-16vcpu-200gb \
    --autoscale-min 0 --autoscale-max 4 \
    --disk-size 500Gi

# Install Knative Serving (official install, one namespace)
kubectl apply -f https://github.com/knative/serving/releases/latest/download/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/latest/download/serving-core.yaml
kubectl apply -f https://github.com/knative/net-kourier/releases/latest/download/kourier.yaml
kubectl patch configmap/config-network --namespace knative-serving --type merge \
    -p '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml
```

### C.2 Deploy

```bash
kubectl create namespace textworld-vr
# fill in IMAGE + S3 vars, then apply:
envsubst < textworld-vr/deploy/k8s/knative.yaml | kubectl apply -f -

# Get the URL
kubectl get ksvc textworld-vr -n textworld-vr -o jsonpath='{.status.url}'
```

Knative annotations in the manifest (`deploy/k8s/knative.yaml`):
- `autoscaling.knative.dev/min-scale: "0"` — scale to zero when idle
- `autoscaling.knative.dev/max-scale: "4"` — cap parallel generations
- `autoscaling.knative.dev/target: "1"` — one job per pod
- `autoscaling.knative.dev/scale-to-zero-grace-period: "5m"` — idle period before scale-down
- `containerConcurrency: 1` — never two jobs on one pod
- `timeoutSeconds: 3600` — long-poll the generate endpoint

### C.3 Cost

- GPU node pool autoscales to 0 when no pods — you pay for nothing between jobs
- First request after a cold period waits ~3 min for pod boot
- Warm pods serve requests instantly until 5 min idle → scale down

---

## 2. Observability

- Ray dashboard on `:8265` (Path A/B) or via `kubectl port-forward` (Path C)
- App logs via `nebius ai endpoint logs --name textworld-vr` (Path B) or `kubectl logs -n textworld-vr -l serving.knative.dev/service=textworld-vr` (Path C)
- `/idle` endpoint reports seconds since last activity + active job count

## 3. Troubleshooting

| Symptom | Fix |
|---|---|
| `HF_TOKEN not set` at boot | Accept FLUX license on huggingface, set `HF_TOKEN` in secrets |
| FLUX-Fill 403 on download | License not accepted for FLUX.1-Fill-dev specifically (separate repo) |
| `checkpoints/depth_anything_v2_vitl.pth not found` | `stage_models.py` failed — check logs, re-run |
| OOM during `gen_layerdata` | Drop concurrency (Knative: `containerConcurrency: 1`) and retry |
| Cold start >10 min | Pre-warm the model PVC; first-ever boot downloads 80 GB |
| Generated `.ply` shows black in browser | 2 GB .ply stresses the mkkellogg parser; convert to .ksplat (see `TODO-ksplat`) |

## 4. What's NOT included (yet)

- `.ksplat` conversion (requires Node/npm; planned as a post-processing step)
- HTTPS custom domain (use Knative's DomainMapping or put Cloudflare in front)
- Real queue (current impl is in-memory; use Redis/NATS for horizontal scale-out)
- Per-user auth (single bearer token today; add JWT/OIDC for multi-tenant)
