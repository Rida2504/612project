#!/bin/bash
# stop_when_idle.sh — idle watchdog that stops the Nebius endpoint when no jobs
# have been in flight for $IDLE_MINUTES consecutive minutes.
#
# Run this as a sidecar (CronJob on MK8s) OR from your laptop via cron.
# Requires: `nebius` CLI authenticated, jq.
#
# To wake the endpoint back up, pair this with a tiny CPU endpoint that runs
# start.sh on demand (see wake-webhook/ if you need one), or do it manually.

set -euo pipefail

: "${PARENT_ID:?PARENT_ID required}"
ENDPOINT_NAME="${ENDPOINT_NAME:-textworld-vr}"
ENDPOINT_URL="${ENDPOINT_URL:?ENDPOINT_URL required (public https url)}"
AUTH_TOKEN="${ENDPOINT_AUTH_TOKEN:?ENDPOINT_AUTH_TOKEN required}"
IDLE_MINUTES="${IDLE_MINUTES:-15}"

# Query the server for its last-activity timestamp. Our API exposes /idle on
# localhost:8000 that returns JSON {last_job_at_s_ago: float}.
idle_seconds=$(curl -fsSL -H "Authorization: Bearer $AUTH_TOKEN" \
    "$ENDPOINT_URL/idle" | jq -r '.last_activity_seconds_ago // 0')
idle_s=${idle_seconds%.*}
echo "[$(date -Is)] idle_seconds=$idle_s (threshold=$((IDLE_MINUTES*60)))"

if [ "$idle_s" -gt "$((IDLE_MINUTES*60))" ]; then
    echo "idle ≥ ${IDLE_MINUTES}m → stopping endpoint"
    nebius ai endpoint stop --name "$ENDPOINT_NAME" --parent-id "$PARENT_ID"
    echo "   stopped. Re-start with: nebius ai endpoint start --name $ENDPOINT_NAME --parent-id $PARENT_ID"
else
    echo "   still active — not stopping"
fi
