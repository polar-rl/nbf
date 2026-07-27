#!/bin/bash
#SBATCH --job-name=l96-grid-merge
#SBATCH --account=aip-nathanst
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/solinas/scratch/logs/l96_grid_merge_%j.out
#SBATCH --mail-user=solinas@ualberta.ca
#SBATCH --mail-type=FAIL,END
#
# Reduce step for the L96 grid array job: reads OUT_DIR/_shards/*.npz and writes
# {cell}.npz plus the summary/tex tables. No GPU needed.
#
# Submit chained to the array job so it runs only if every task succeeded:
#   sbatch --dependency=afterok:<array_job_id> scripts/eval_l96_grid_merge_slurm.sh
#
# SEEDS and METHODS must match the array job: SEEDS sets the expected shard count
# (a warning names any that are missing) and METHODS sets the table row order.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/solinas/projects/aip-nathanst/solinas/polar-rl/nbf}
SCRATCH_BASE=${SCRATCH_BASE:-/home/solinas/scratch/nbf/l96}

OUT_DIR="${SCRATCH_BASE}/results/k20/grid"
SEEDS=${SEEDS:-0,1,2,3,4}

cd "${PROJECT_DIR}"

module load cuda
unset LD_LIBRARY_PATH
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

ARGS=(
    --merge
    --output_dir="${OUT_DIR}"
    --seeds="${SEEDS}"
)
if [ -n "${METHODS:-}" ]; then
    ARGS+=(--methods="${METHODS}")
fi

python3 -m nbf.domains.lorenz96.run_grid "${ARGS[@]}"
