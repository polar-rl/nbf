#!/usr/bin/env bash
# Pursuit-evasion smoke / integration test (the PE regression gate).
#
# Trains a tiny k=1 NBF on-policy (episodes simulated on the fly under training-mixture
# policies, no PF corpus) and checks the checkpoint artefacts the evaluator needs. The
# evaluation step is added with the PE grid runner (nbf.domains.pursuit_evasion.run_grid).
# A few CPU minutes; not a correctness test.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Use the project venv when invoked outside `uv run`.
if command -v python >/dev/null 2>&1; then PY=python; else PY="uv run python"; fi

CONFIG="nbf/domains/pursuit_evasion/configs/smoke_k1.yaml"
MODEL_DIR="results/pursuit_evasion/smoke"

echo "=== [1/1] Train (on-policy, no corpus) ==="
$PY -m nbf.domains.pursuit_evasion.train --config "$CONFIG" --set "save_path=$MODEL_DIR/"

for f in model.eqx model_config.json data_config.json train_config.json train_log.jsonl; do
  test -f "$MODEL_DIR/$f" || { echo "FAIL: missing $MODEL_DIR/$f"; exit 1; }
done
tail -n 1 "$MODEL_DIR/train_log.jsonl"

echo "=== Smoke test PASS ==="
