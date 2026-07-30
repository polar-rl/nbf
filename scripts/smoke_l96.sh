#!/usr/bin/env bash
# Lorenz-96 smoke / integration test.
#
# Runs the full L96 pipeline against a tiny config: gold-SMC datagen, a small
# NBF model, then a tiny MMD-vs-oracle evaluation. Designed to finish in a few
# minutes on CPU. Not a correctness test — the model is too small to produce
# meaningful metrics; this just exercises every code path end-to-end.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="nbf/domains/lorenz96/configs/smoke_l96.yaml"
MODEL_DIR="results/lorenz96/smoke/nbf"
OUT_DIR="results/lorenz96/smoke/eval"

export NBF_GOLD_SMC_COUNT=1024
export NBF_NUM_REF_SAMPLES=64

echo "=== [1/3] Generate data (gold SMC) ==="
python -m nbf.domains.lorenz96.generate_data --config "$CONFIG"

echo "=== [2/3] Train ==="
python -m nbf.domains.lorenz96.train --config "$CONFIG"

test -f "$MODEL_DIR/model.eqx" || { echo "FAIL: missing $MODEL_DIR/model.eqx"; exit 1; }
test -f "$MODEL_DIR/model_config.json" || { echo "FAIL: missing $MODEL_DIR/model_config.json"; exit 1; }
test -f "$MODEL_DIR/data_config.json" || { echo "FAIL: missing $MODEL_DIR/data_config.json"; exit 1; }

echo "=== [3/3] Evaluate (tiny method set) ==="
python -m nbf.domains.lorenz96.evaluate \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUT_DIR" \
    --num_episodes 4 \
    --episode_length 5 \
    --methods nbf_64,pf_256

test -f "$OUT_DIR/l96_mmd.npz" || { echo "FAIL: missing $OUT_DIR/l96_mmd.npz"; exit 1; }

echo "=== Smoke test PASS ==="
