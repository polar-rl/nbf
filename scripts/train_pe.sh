#!/usr/bin/env bash
# Local PE training.
#
#   bash scripts/train_pe.sh <k>
#
# Reads nbf/domains/pursuit_evasion/configs/nbf_k${K}.yaml. Generates the PF
# dataset if it's not already cached. Saves the trained model under the path
# the yaml's ``save_path`` resolves to (default: results/pursuit_evasion/k${K}/nbf).
set -euo pipefail

K=${1:?usage: $0 <k>  (e.g. bash $0 1)}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="nbf/domains/pursuit_evasion/configs/nbf_k${K}.yaml"
[ -f "$CONFIG" ] || { echo "no config at $CONFIG"; exit 1; }

python -m nbf.domains.pursuit_evasion.train --config "$CONFIG" "${@:2}"
