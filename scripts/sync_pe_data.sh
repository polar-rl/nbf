#!/bin/bash
# Sync PE belief data from the cluster's scratch to local data/pursuit_evasion/.
# Layout: pe_belief_data_k{K}/{obs,particles_t,particles_tp1}.npy
#
#   bash scripts/sync_pe_data.sh
#
# Tunables (env vars):
#   REMOTE        ssh target, default solinas@vulcan.alliancecan.ca
#   REMOTE_DIR    scratch data directory on the cluster
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE=${REMOTE:-solinas@vulcan.alliancecan.ca}
REMOTE_DIR=${REMOTE_DIR:-/home/solinas/scratch/nbf/pe/data/}
LOCAL_DIR=${LOCAL_DIR:-data/pursuit_evasion/}

mkdir -p "$LOCAL_DIR"

rsync -avzP \
    --include='pe_belief_data_k*/' \
    --include='pe_belief_data_k*/*.npy' \
    --exclude='*' \
    "${REMOTE}:${REMOTE_DIR}" "$LOCAL_DIR"
