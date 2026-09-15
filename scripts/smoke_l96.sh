#!/usr/bin/env bash
# Lorenz-96 smoke / integration test.
#
# Runs the L96 pipeline against a tiny config: on-policy gold-state training of a
# small NBF model, then a tiny reference-free evaluation grid (NLL, RMSE, CRPS,
# coverage, spread). Designed to finish in a few minutes on CPU. Not a correctness
# test: the model is too small to produce meaningful metrics; this just exercises
# every code path end-to-end.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Use the project venv when invoked outside `uv run`.
if command -v python >/dev/null 2>&1; then PY=python; else PY="uv run python"; fi

CONFIG="nbf/domains/lorenz96/configs/smoke_l96.yaml"
MODEL_DIR="results/lorenz96/smoke"
OUT_DIR="results/lorenz96/smoke/eval"

export NBF_NUM_REF_SAMPLES=64

echo "=== [1/2] Train (on-policy, no corpus) ==="
$PY -m nbf.domains.lorenz96.train --config "$CONFIG" --set "save_path=$MODEL_DIR/"

test -f "$MODEL_DIR/model.eqx" || { echo "FAIL: missing $MODEL_DIR/model.eqx"; exit 1; }
test -f "$MODEL_DIR/model_config.json" || { echo "FAIL: missing $MODEL_DIR/model_config.json"; exit 1; }
test -f "$MODEL_DIR/data_config.json" || { echo "FAIL: missing $MODEL_DIR/data_config.json"; exit 1; }

echo "=== [2/2] Evaluate (L2 cell, tiny method set) ==="
$PY -m nbf.domains.lorenz96.run_grid l2 \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUT_DIR" \
    --num_episodes 4 \
    --episode_length 5 \
    --seeds 0 \
    --methods nbf_64,nbf_op_64,pf_256,pf_reg_256,enkf_20

test -f "$OUT_DIR/l2.npz" || { echo "FAIL: missing $OUT_DIR/l2.npz"; exit 1; }
test -f "$OUT_DIR/l2_summary.md" || { echo "FAIL: missing $OUT_DIR/l2_summary.md"; exit 1; }
cat "$OUT_DIR/l2_summary.md"

echo "=== Smoke test PASS ==="
