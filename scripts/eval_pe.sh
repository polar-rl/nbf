#!/usr/bin/env bash
# Local PE evaluation. Reads trained models from results/pursuit_evasion/k${K}/nbf/.
#
#   bash scripts/eval_pe.sh
#
# Tunables (env vars):
#   NUM_EPISODES, EPISODE_LENGTH, NUM_SEEDS
#   RESULTS               base directory holding k*/nbf/ subdirs
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS=${RESULTS:-results/pursuit_evasion}
MODELS="1:${RESULTS}/k1/nbf,2:${RESULTS}/k2/nbf,3:${RESULTS}/k3/nbf,5:${RESULTS}/k5/nbf"

NUM_EPISODES=${NUM_EPISODES:-50}
EPISODE_LENGTH=${EPISODE_LENGTH:-15}
NUM_SEEDS=${NUM_SEEDS:-3}

run() {
    local exps="$1" model_dir="$2" out_dir="$3"; shift 3
    mkdir -p "${out_dir}"
    python -m nbf.domains.pursuit_evasion.evaluate \
        --experiments="${exps}" \
        --model_dir="${model_dir}" \
        --model_dirs_per_k="${MODELS}" \
        --output_dir="${out_dir}" \
        --num_episodes="${NUM_EPISODES}" \
        --episode_length="${EPISODE_LENGTH}" \
        --num_seeds="${NUM_SEEDS}" \
        "$@"
}

# E1: k=1 heatmaps. Uncomment when you want it.
#run 1 "${RESULTS}/k1/nbf" "${RESULTS}/k1/eval"

# E2: cross-k MMD + per-policy plots. At k=1 only for the local run.
run 2 "${RESULTS}/k1/nbf" "${RESULTS}/cross_k/eval" \
    --e2_ks=1

# E4: single-k panel at k=1.
run 4 "${RESULTS}/k1/nbf" "${RESULTS}/k1/eval"

# E5: single-k panel at k=3. Uncomment when you have a k=3 model.
#run 5 "${RESULTS}/k3/nbf" "${RESULTS}/k3/eval"
