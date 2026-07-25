#!/bin/bash
#SBATCH --job-name=l96-grid
#SBATCH --account=aip-nathanst
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gpus-per-task=1
#SBATCH --array=1,2,3
#SBATCH --requeue
#SBATCH --output=/home/solinas/scratch/logs/l96_grid_l%a_%j.out
#SBATCH --mail-user=solinas@ualberta.ca
#SBATCH --mail-type=FAIL,REQUEUE,END
#
# SLURM batch script for the L96 experiment grid. Array task index = cell:
#   1 -> L1 (in-distribution F in {6, 10})
#   2 -> L2 (zero-shot forcing F = 8, the headline)
#   3 -> L3 (zero-shot sigma_obs sweep at F = 8)
#
# Tunables (env vars):
#   PROJECT_DIR    repo checkout on the cluster
#   SCRATCH_BASE   scratch root containing results/
#   METHODS, NUM_EPISODES, EPISODE_LENGTH, SEEDS
set -euo pipefail

CELL="l${SLURM_ARRAY_TASK_ID}"
PROJECT_DIR=${PROJECT_DIR:-/home/solinas/projects/aip-nathanst/solinas/polar-rl/nbf}
SCRATCH_BASE=${SCRATCH_BASE:-/home/solinas/scratch/nbf/l96}

MODEL_DIR="${SCRATCH_BASE}/results/k20/nbf"
OUT_DIR="${SCRATCH_BASE}/results/k20/grid"

NUM_EPISODES=${NUM_EPISODES:-100}
EPISODE_LENGTH=${EPISODE_LENGTH:-0}
SEEDS=${SEEDS:-0,1,2,3,4}

mkdir -p "${OUT_DIR}" "/home/solinas/scratch/logs"

cd "${PROJECT_DIR}"

module load cuda
unset LD_LIBRARY_PATH
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

echo "Cell: ${CELL}"
echo "Model dir: ${MODEL_DIR}"
echo "Output dir: ${OUT_DIR}"

ARGS=(
    "${CELL}"
    --model_dir="${MODEL_DIR}"
    --output_dir="${OUT_DIR}"
    --num_episodes="${NUM_EPISODES}"
    --episode_length="${EPISODE_LENGTH}"
    --seeds="${SEEDS}"
)
if [ -n "${METHODS:-}" ]; then
    ARGS+=(--methods="${METHODS}")
fi

python3 -m nbf.domains.lorenz96.run_grid "${ARGS[@]}"
