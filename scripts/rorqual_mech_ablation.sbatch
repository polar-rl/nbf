#!/bin/bash
#SBATCH --job-name=l96-mech
#SBATCH --account=rrg-mbowling-ad
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#
# Within-step mechanism ablation (Uymn Q1): weighted set vs fit-and-regenerate
# at a fixed particle budget, across likelihood concentrations, scored against
# the gold cloud's own updated posterior. Sweep stays INSIDE the training range
# [0.5, 1.1] so every cell is in-family.
#
# Required env:
#   MODEL_DIR  trained checkpoint dir (model.eqx + model_config.json + data_config.json)
# Optional env:
#   OUT_DIR, SIGMAS, SEEDS, NUM_EPISODES, NUM_GOLD, BUDGETS, CHUNK_SIZE, FM_NUM_STEPS
set -euo pipefail
: "${SLURM_TMPDIR:?SLURM_TMPDIR unset}"
: "${MODEL_DIR:?set MODEL_DIR to a trained checkpoint dir}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
SCRATCH=${SCRATCH:-/scratch/$USER}

BASE=$HOME/projects/rrg-mbowling-ad/solinas
export UV_CACHE_DIR=$BASE/uv-cache
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT=$SLURM_TMPDIR/venv

OUT_DIR=${OUT_DIR:-$SCRATCH/nbf/l96/$(basename "$MODEL_DIR")/mech}
SIGMAS=${SIGMAS:-0.5,0.625,0.75,0.875,1.0}
SEEDS=${SEEDS:-0,1,2}
NUM_EPISODES=${NUM_EPISODES:-100}
NUM_GOLD=${NUM_GOLD:-2000000}
BUDGETS=${BUDGETS:-1024}
CHUNK_SIZE=${CHUNK_SIZE:-4}
FM_NUM_STEPS=${FM_NUM_STEPS:-100}

mkdir -p "$OUT_DIR"

cd "$BASE/repos/nbf"
uv sync --frozen --extra gpu
PY=$SLURM_TMPDIR/venv/bin/python

for r in nbf embedex; do
  echo "$r $(git -C "$BASE/repos/$r" rev-parse HEAD)"
done > "$OUT_DIR/git-shas.txt"

echo "model=$MODEL_DIR out=$OUT_DIR"
echo "sigmas=$SIGMAS seeds=$SEEDS gold=$NUM_GOLD budgets=$BUDGETS fm_steps=$FM_NUM_STEPS"

$PY -m nbf.domains.lorenz96.run_mechanism_ablation \
    --model_dir="$MODEL_DIR" \
    --output_dir="$OUT_DIR" \
    --sigmas="$SIGMAS" \
    --budgets="$BUDGETS" \
    --seeds="$SEEDS" \
    --num_episodes="$NUM_EPISODES" \
    --num_gold="$NUM_GOLD" \
    --chunk_size="$CHUNK_SIZE" \
    --fm_num_steps="$FM_NUM_STEPS"
echo MECH-OK
