#!/bin/bash
#SBATCH --job-name=l96-grid
#SBATCH --account=rrg-mbowling-ad
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=3:00:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH --output=slurm-%x-t%a-%j.out
#
# L96 eval grid on rorqual. One array task = one (cell, subcell, seed) shard,
# owning one gold-SMC oracle. With l3 removed from CELLS and SEEDS=0,1,2,3,4:
#   l1 (f6, f10)  2 subcells x 5 seeds = 10  -> tasks 0-9
#   l2 (f8)       1 subcell  x 5 seeds =  5  -> tasks 10-14
# Confirm with: python -m nbf.domains.lorenz96.run_grid --list_tasks --output_dir=/tmp/x
# After the array: merge with --merge --t_min 5 (same SEEDS!) on any node or
# the workstation once shards are rsynced back.
#
# Required env:
#   MODEL_DIR  directory with model.eqx + model_config.json + data_config.json
# Optional env:
#   OUT_DIR, METHODS, NUM_EPISODES, SEEDS, GOLD_SMC_COUNT, CHUNK_SIZE, TIMING_REPEATS
set -euo pipefail
: "${SLURM_TMPDIR:?SLURM_TMPDIR unset}"
: "${MODEL_DIR:?set MODEL_DIR to a trained checkpoint dir}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
SCRATCH=${SCRATCH:-/scratch/$USER}

BASE=$HOME/projects/rrg-mbowling-ad/solinas
export UV_CACHE_DIR=$BASE/uv-cache
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT=$SLURM_TMPDIR/venv

OUT_DIR=${OUT_DIR:-$SCRATCH/nbf/l96/$(basename "$MODEL_DIR")/grid}
METHODS=${METHODS:-nbf_1024,nbf_2048,nbf_op_1024,nbf_op_2048,pf_16384,pf_32768,pf_reg_16384,pf_reg_32768,pf_rm_16384,pf_rm_32768,pf_op_16384,pf_op_32768,enkf_50,enkf_100,pf_oracle}
NUM_EPISODES=${NUM_EPISODES:-100}
SEEDS=${SEEDS:-0,1,2,3,4}
GOLD_SMC_COUNT=${GOLD_SMC_COUNT:-200000}
CHUNK_SIZE=${CHUNK_SIZE:-16}
TIMING_REPEATS=${TIMING_REPEATS:-0}

mkdir -p "$OUT_DIR"

cd "$BASE/repos/nbf"
uv sync --frozen --extra gpu
PY=$SLURM_TMPDIR/venv/bin/python
export NBF_GOLD_SMC_COUNT="$GOLD_SMC_COUNT"

for r in nbf embedex; do
  echo "$r $(git -C "$BASE/repos/$r" rev-parse HEAD)"
done > "$OUT_DIR/git-shas.txt"

echo "Task $SLURM_ARRAY_TASK_ID  model=$MODEL_DIR  out=$OUT_DIR  methods=$METHODS"
$PY -m nbf.domains.lorenz96.run_grid \
    --task_id="$SLURM_ARRAY_TASK_ID" \
    --model_dir="$MODEL_DIR" \
    --output_dir="$OUT_DIR" \
    --methods="$METHODS" \
    --num_episodes="$NUM_EPISODES" \
    --seeds="$SEEDS" \
    --chunk_size="$CHUNK_SIZE" \
    --timing_repeats="$TIMING_REPEATS"
echo GRID-TASK-OK
