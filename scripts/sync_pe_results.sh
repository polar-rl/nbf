#!/bin/bash
# Sync PE model + eval results from the cluster's scratch to local results/pursuit_evasion/.
#
#   bash scripts/sync_pe_results.sh
#
# Tunables (env vars):
#   REMOTE        ssh target, default solinas@vulcan.alliancecan.ca
#   REMOTE_DIR    scratch results directory on the cluster
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE=${REMOTE:-solinas@vulcan.alliancecan.ca}
REMOTE_DIR=${REMOTE_DIR:-/home/solinas/scratch/nbf/pe/results/}
LOCAL_DIR=${LOCAL_DIR:-results/pursuit_evasion/}

mkdir -p "$LOCAL_DIR"

rsync -avz --include='*/' \
    --include='*.eqx' \
    --include='*.json' \
    --include='*.pdf' \
    --include='*.npz' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_DIR}" "$LOCAL_DIR"
