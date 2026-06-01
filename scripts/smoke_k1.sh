#!/usr/bin/env bash
# k=1 smoke / integration test.
#
# Runs the full PE pipeline against a tiny config: trains a small NBF model,
# then runs every remaining experiment (E1..E5). Designed to finish
# in a few minutes on CPU. Not a correctness test — the model is too small to
# produce meaningful metrics; this just exercises every code path end-to-end.
#
# The NBF_* env vars below shrink the gold-SMC oracle and reference/negative
# pools so the heavy oracle generation per scenario stays cheap. Production
# defaults are still 100_000 / 1024 / 256 — see eval/dispatch.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="nbf/domains/pursuit_evasion/configs/smoke_k1.yaml"
MODEL_DIR="results/pursuit_evasion/smoke/k1/nbf"
OUT_DIR="results/pursuit_evasion/smoke/k1/eval"

export NBF_GOLD_SMC_COUNT=1024
export NBF_ORACLE_PF_COUNT=1024
export NBF_NUM_REF_SAMPLES=64
export NBF_NUM_NEGATIVES=32

echo "=== [1/2] Train (auto-generates data if missing) ==="
python -m nbf.domains.pursuit_evasion.train --config "$CONFIG"

# Sanity check: the trainer produced the artefacts the evaluator needs.
test -f "$MODEL_DIR/model.eqx" || { echo "FAIL: missing $MODEL_DIR/model.eqx"; exit 1; }
test -f "$MODEL_DIR/model_config.json" || { echo "FAIL: missing $MODEL_DIR/model_config.json"; exit 1; }
test -f "$MODEL_DIR/data_config.json" || { echo "FAIL: missing $MODEL_DIR/data_config.json"; exit 1; }

echo "=== [2/2] Evaluate E1..E5 (tiny method sets) ==="
python -m nbf.domains.pursuit_evasion.evaluate \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUT_DIR" \
    --experiments 1,2,3,4,5 \
    --num_episodes 4 \
    --episode_length 5 \
    --e1_methods nbf_64,pf_256 \
    --e2_ks 1 \
    --e2_methods nbf_64,pf_256 \
    --e3_methods nbf_64,pf_256 \
    --e4_sigma_methods nbf_64,pf_256 \
    --e4_sigmas 0.1,0.5 \
    --e5_methods nbf_64,pf_256 \
    --e5_num_episodes 1 \
    --e5_num_warmup 0 \
    --e5_num_repeats 1

# Sanity check: each experiment wrote *something* to the output directory.
test -n "$(ls -A "$OUT_DIR" 2>/dev/null || true)" \
    || { echo "FAIL: $OUT_DIR is empty after eval"; exit 1; }

echo "=== Smoke test PASS ==="
