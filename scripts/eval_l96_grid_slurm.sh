#!/bin/bash
#SBATCH --job-name=l96-grid
#SBATCH --account=aip-nathanst
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gpus-per-task=1
#SBATCH --array=0-44
#SBATCH --requeue
#SBATCH --output=/home/solinas/scratch/logs/l96_grid_t%a_%j.out
#SBATCH --mail-user=solinas@ualberta.ca
#SBATCH --mail-type=FAIL,REQUEUE,END
#
# SLURM array job for the L96 experiment grid. Array task index = one
# (cell, subcell, seed) unit -- the scope that owns exactly one gold-SMC oracle,
# so no oracle is ever computed twice. With the default SEEDS that is 45 tasks:
#   l1 (F in {6, 10})        2 subcells x 5 seeds = 10   -> tasks 0-9
#   l2 (zero-shot F = 8)     1 subcell  x 5 seeds =  5   -> tasks 10-14
#   l3 (sigma_obs sweep)     6 subcells x 5 seeds = 30   -> tasks 15-44
# Confirm the mapping for your settings with:
#   python3 -m nbf.domains.lorenz96.run_grid --list_tasks --output_dir=OUT
#
# Each task writes OUT_DIR/_shards/{cell}__{subcell}__s{seed}.npz and nothing else,
# so tasks never race on a shared output file. Once the array finishes, reduce the
# shards into {cell}.npz + the summary/tex tables:
#   sbatch --dependency=afterok:<array_job_id> scripts/eval_l96_grid_merge_slurm.sh
#
# IMPORTANT: SEEDS must be identical for the array and the merge -- it determines
# the task numbering and the expected shard count.
#
# Tunables (env vars):
#   PROJECT_DIR    repo checkout on the cluster
#   SCRATCH_BASE   scratch root containing results/
#   METHODS, NUM_EPISODES, EPISODE_LENGTH, SEEDS, GOLD_SMC_COUNT, CHUNK_SIZE
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/solinas/projects/aip-nathanst/solinas/polar-rl/nbf}
SCRATCH_BASE=${SCRATCH_BASE:-/home/solinas/scratch/nbf/l96}

MODEL_DIR="${SCRATCH_BASE}/results/k20/nbf"
OUT_DIR="${SCRATCH_BASE}/results/k20/grid"

NUM_EPISODES=${NUM_EPISODES:-100}
EPISODE_LENGTH=${EPISODE_LENGTH:-0}
SEEDS=${SEEDS:-0,1,2,3,4}
GOLD_SMC_COUNT=${GOLD_SMC_COUNT:-200000}
CHUNK_SIZE=${CHUNK_SIZE:-4}

mkdir -p "${OUT_DIR}" "/home/solinas/scratch/logs"

cd "${PROJECT_DIR}"

module load cuda
unset LD_LIBRARY_PATH
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export NBF_GOLD_SMC_COUNT="${GOLD_SMC_COUNT}"

echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Gold-SMC oracle particles: ${NBF_GOLD_SMC_COUNT}"
echo "Episode chunk size: ${CHUNK_SIZE}"
echo "Model dir: ${MODEL_DIR}"
echo "Output dir: ${OUT_DIR}"

ARGS=(
    --task_id="${SLURM_ARRAY_TASK_ID}"
    --model_dir="${MODEL_DIR}"
    --output_dir="${OUT_DIR}"
    --num_episodes="${NUM_EPISODES}"
    --episode_length="${EPISODE_LENGTH}"
    --seeds="${SEEDS}"
    --chunk_size="${CHUNK_SIZE}"
)
if [ -n "${METHODS:-}" ]; then
    ARGS+=(--methods="${METHODS}")
fi

python3 -m nbf.domains.lorenz96.run_grid "${ARGS[@]}"
