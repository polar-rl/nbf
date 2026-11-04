#!/usr/bin/env bash
# Pursuit-evasion smoke / integration test.
#
# Trains a tiny k=1 NBF on-policy (episodes simulated on the fly under training-mixture
# policies), then runs the reference-free PE grid on one cell with a tiny
# method set. A few CPU minutes; not a correctness test.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Use the project venv when invoked outside `uv run`.
if command -v python >/dev/null 2>&1; then PY=python; else PY="uv run python"; fi

CONFIG="nbf/domains/pursuit_evasion/configs/smoke_k1.yaml"
MODEL_DIR="results/pursuit_evasion/smoke"
OUT_DIR="results/pursuit_evasion/smoke/eval"

export NBF_NUM_REF_SAMPLES=64

echo "=== [1/2] Train (on-policy) ==="
$PY -m nbf.domains.pursuit_evasion.train --config "$CONFIG" --set "save_path=$MODEL_DIR/"

for f in model.eqx model_config.json data_config.json train_config.json train_log.jsonl; do
  test -f "$MODEL_DIR/$f" || { echo "FAIL: missing $MODEL_DIR/$f"; exit 1; }
done
tail -n 1 "$MODEL_DIR/train_log.jsonl"

echo "=== [2/2] Evaluate (ood cell, tiny method set) ==="
$PY -m nbf.domains.pursuit_evasion.run_grid ood \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUT_DIR" \
    --num_episodes 4 \
    --episode_length 5 \
    --seeds 0 \
    --methods nbf_64,pf_256,pf_reg_256,gmf_2

test -f "$OUT_DIR/ood.npz" || { echo "FAIL: missing $OUT_DIR/ood.npz"; exit 1; }
cat "$OUT_DIR/ood_summary.md"

echo "=== Smoke test PASS ==="
