#!/bin/bash
# Deploy TextWorld VR to Nebius Serverless AI Endpoints.
#
# What this does:
#   1. Build + push image to Nebius Container Registry.
#   2. Create MysteryBox secrets for HF_TOKEN, endpoint auth token, and S3 creds.
#   3. Create (or recreate) the endpoint: H200 SXM (us-central1 default; H100 SXM
#      in eu-north1), public HTTPS, token auth, env vars wired to the secrets.
#   4. Print the invoke URL and a smoke-test curl.
#
# Scale-to-zero:
#   Nebius Serverless AI endpoints do not currently expose autoscaling flags
#   (confirmed: no --min-replicas / --max-replicas / --idle-timeout on
#   `nebius ai endpoint create`). Lifecycle is manual: start / stop.
#   The companion stop_when_idle.sh polls /idle and stops when idle >15 min.
#   For request-driven 0→N autoscale see ../k8s/knative.yaml.
#
# Prereqs:
#   - Nebius CLI installed + `nebius profile create` configured
#   - HF token with FLUX.1-dev AND FLUX.1-Fill-dev licenses accepted
#   - S3 bucket + access keys in Nebius Object Storage
#
# Required env vars:
#   PARENT_ID              Nebius project/folder id (iam container)
#   REGISTRY_ID            Container Registry id
#   SUBNET_ID              VPC subnet id
#   HF_TOKEN               HuggingFace token
#   S3_BUCKET              scenes bucket name
#   S3_ACCESS_KEY_ID       + S3_SECRET_ACCESS_KEY
#
# Optional env vars:
#   IMAGE_TAG              defaults to short git sha or "dev"
#   ENDPOINT_NAME          defaults to textworld-vr
#   NEBIUS_REGION          defaults to us-central1 (H100 SXM needs eu-north1)
#   PLATFORM               defaults to gpu-h200-sxm (or gpu-h100-sxm in eu-north1)
#   PRESET                 defaults to 1gpu-16vcpu-200gb
#   S3_ENDPOINT_URL        defaults to https://storage.<NEBIUS_REGION>.nebius.cloud
#   S3_PREFIX              defaults to scenes/
#   S3_PUBLIC_BASE_URL     if set, viewer uses direct public URLs instead of presigned

set -euo pipefail

: "${PARENT_ID:?PARENT_ID required}"
: "${REGISTRY_ID:?REGISTRY_ID required}"
: "${SUBNET_ID:?SUBNET_ID required}"
: "${HF_TOKEN:?HF_TOKEN required}"
: "${S3_BUCKET:?S3_BUCKET required}"
: "${S3_ACCESS_KEY_ID:?S3_ACCESS_KEY_ID required}"
: "${S3_SECRET_ACCESS_KEY:?S3_SECRET_ACCESS_KEY required}"

IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"
ENDPOINT_NAME="${ENDPOINT_NAME:-textworld-vr}"
NEBIUS_REGION="${NEBIUS_REGION:-us-central1}"
# H100 SXM is eu-north1 only; H200 SXM is available in us-central1 + other regions.
# Default matches Nebius CLI default ("gpu-h100-sxm in eu-north1 and gpu-h200-sxm elsewhere").
if [ "$NEBIUS_REGION" = "eu-north1" ]; then
    PLATFORM="${PLATFORM:-gpu-h100-sxm}"
else
    PLATFORM="${PLATFORM:-gpu-h200-sxm}"
fi
PRESET="${PRESET:-1gpu-16vcpu-200gb}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://storage.${NEBIUS_REGION}.nebius.cloud}"
S3_PREFIX="${S3_PREFIX:-scenes/}"
ENDPOINT_AUTH_TOKEN="${ENDPOINT_AUTH_TOKEN:-$(uuidgen | tr -d '-' | head -c 32)}"

# Container Registry hostname follows cr.<region>.nebius.cloud (not .nebiuscloud.net).
# The registry PATH in the image URL is the ID's second field only (Nebius quickstart
# uses `cut -d- -f 2`): `registry-u00w6p...xpdn` → `u00w6p...xpdn`.
REGISTRY_HOST="cr.${NEBIUS_REGION}.nebius.cloud"
REGISTRY_PATH="${REGISTRY_ID#registry-}"
IMAGE_REMOTE="${REGISTRY_HOST}/${REGISTRY_PATH}/textworld-vr:${IMAGE_TAG}"

echo "== TextWorld VR → Nebius"
echo "   image    : $IMAGE_REMOTE"
echo "   endpoint : $ENDPOINT_NAME  ($PLATFORM / $PRESET)"
echo "   S3       : s3://$S3_BUCKET/$S3_PREFIX ($S3_ENDPOINT_URL)"

# ---------------------------------------------------------------------------
# 1. Build + push
# ---------------------------------------------------------------------------
echo "-- docker build --"
docker build --platform=linux/amd64 -t "$IMAGE_REMOTE" -f textworld-vr/deploy/Dockerfile .
echo "-- docker push --"
docker push "$IMAGE_REMOTE"

# ---------------------------------------------------------------------------
# 2. MysteryBox secrets.
#
# Docs: `nebius mysterybox secret create --parent-id <id> --name <name>
#         --secret-version-payload '[{"key":"X","string_value":"Y"}]'`
#
# The payload KEY must match what the endpoint flag expects:
#   --token-secret <name>            → payload key must be "AUTH_TOKEN"
#   --env-secret HF_TOKEN=<name>     → payload key must be "HF_TOKEN"
# So each secret's payload key == the env var name on the other end.
# ---------------------------------------------------------------------------

create_or_replace_secret() {
    local secret_name="$1"
    local payload_key="$2"
    local payload_value="$3"
    local payload_json
    payload_json=$(jq -nc --arg k "$payload_key" --arg v "$payload_value" \
        '[{key:$k, string_value:$v}]')

    # If it exists, delete (we always rewrite). Nebius MysteryBox supports
    # add-version but the simpler flow for deploy is create-on-clean-slate.
    local existing
    existing=$(nebius mysterybox secret get-by-name \
        --name "$secret_name" --parent-id "$PARENT_ID" --format json 2>/dev/null \
        | jq -r '.metadata.id // empty')
    if [ -n "$existing" ]; then
        nebius mysterybox secret delete --id "$existing" >/dev/null 2>&1 || true
    fi

    nebius mysterybox secret create \
        --parent-id "$PARENT_ID" \
        --name "$secret_name" \
        --secret-version-payload "$payload_json" >/dev/null
    echo "   secret: $secret_name (key=$payload_key)"
}

# Each secret has ONE payload entry whose key is set to match the flag binding.
create_or_replace_secret "tvr-hf-token"              "HF_TOKEN"              "$HF_TOKEN"
create_or_replace_secret "tvr-endpoint-auth-token"   "AUTH_TOKEN"            "$ENDPOINT_AUTH_TOKEN"
create_or_replace_secret "tvr-s3-access"             "AWS_ACCESS_KEY_ID"     "$S3_ACCESS_KEY_ID"
create_or_replace_secret "tvr-s3-secret"             "AWS_SECRET_ACCESS_KEY" "$S3_SECRET_ACCESS_KEY"

# ---------------------------------------------------------------------------
# 3. Endpoint create (recreate if exists).
#
# `delete` requires --id, so resolve by name first.
# ---------------------------------------------------------------------------

existing_id=$(nebius ai endpoint get-by-name \
    --name "$ENDPOINT_NAME" --parent-id "$PARENT_ID" --format json 2>/dev/null \
    | jq -r '.metadata.id // empty')
if [ -n "$existing_id" ]; then
    echo "-- endpoint exists (id=$existing_id); deleting --"
    nebius ai endpoint delete --id "$existing_id" --format json || true
fi

echo "-- create endpoint --"
# Notes on flag choices (all verified against the CLI reference):
#   --auth token                         attach a bearer-token gate
#   --token-secret <name>                secret payload key AUTH_TOKEN
#   --env-secret KEY=<name>              env var KEY, secret payload key must also be KEY
#   --volume s3://<bucket>:/path:rw      bucket mounted read-write at /path
#     (optional trailing ":<profile>" or ":<profile>@<secret>" for S3 auth)
#   --container-port                     repeat per port (8000 for Ray Serve, 8080 for nginx)
#   --disk-size 500Gi                    holds the 80 GB model cache comfortably
#   --shm-size 32Gi                      FLUX-Fill + LLaVA together push /dev/shm

nebius ai endpoint create \
    --name "$ENDPOINT_NAME" \
    --parent-id "$PARENT_ID" \
    --image "$IMAGE_REMOTE" \
    --platform "$PLATFORM" \
    --preset "$PRESET" \
    --public \
    --auth token \
    --token-secret "tvr-endpoint-auth-token" \
    --subnet-id "$SUBNET_ID" \
    --container-port 8000 --container-port 8080 \
    --disk-size 500Gi \
    --shm-size 32Gi \
    --env "SCENES_BACKEND=s3" \
    --env "S3_BUCKET=$S3_BUCKET" \
    --env "S3_ENDPOINT_URL=$S3_ENDPOINT_URL" \
    --env "S3_PREFIX=$S3_PREFIX" \
    --env "S3_PUBLIC_BASE_URL=${S3_PUBLIC_BASE_URL:-}" \
    --env "AWS_DEFAULT_REGION=$NEBIUS_REGION" \
    --env-secret "HF_TOKEN=tvr-hf-token" \
    --env-secret "AWS_ACCESS_KEY_ID=tvr-s3-access" \
    --env-secret "AWS_SECRET_ACCESS_KEY=tvr-s3-secret" \
    --format json

echo
echo "== done"
echo "   auth token: $ENDPOINT_AUTH_TOKEN"
echo "   (also stored in MysteryBox as tvr-endpoint-auth-token, key AUTH_TOKEN)"
echo
echo "Invoke URL is in the JSON above under .status.* — grab it with:"
echo "   nebius ai endpoint get-by-name --name $ENDPOINT_NAME --parent-id $PARENT_ID --format json \\"
echo "       | jq -r '.status.public_url // .status.url'"
echo
echo "Smoke test:"
echo "   URL=\$(nebius ai endpoint get-by-name --name $ENDPOINT_NAME --parent-id $PARENT_ID --format json | jq -r '.status.public_url // .status.url')"
echo "   curl -X POST \"\$URL/generate\" \\"
echo "        -H \"Authorization: Bearer $ENDPOINT_AUTH_TOKEN\" \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"prompt\":\"a cozy japanese coffee shop\",\"seed\":42}'"
