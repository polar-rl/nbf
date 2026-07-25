#!/bin/bash
#SBATCH --job-name=l96-datagen
#SBATCH --account=aip-nathanst
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-task=1
#SBATCH --requeue
#SBATCH --output=/home/solinas/scratch/logs/l96_datagen_%j.out
#SBATCH --mail-user=solinas@ualberta.ca
#SBATCH --mail-type=FAIL,REQUEUE,END
#
# SLURM batch script for L96 belief-data generation (gold SMC, K=20).
#
# Tunables (env vars):
#   PROJECT_DIR    repo checkout on the cluster
#   SCRATCH_BASE   scratch root for data + logs
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/solinas/projects/aip-nathanst/solinas/polar-rl/nbf}
SCRATCH_BASE=${SCRATCH_BASE:-/home/solinas/scratch/nbf/l96}

CONFIG="nbf/domains/lorenz96/configs/nbf_l96.yaml"
DATA_PATH="${SCRATCH_BASE}/data/"

mkdir -p "${DATA_PATH}" "/home/solinas/scratch/logs"

cd "${PROJECT_DIR}"

module load cuda
unset LD_LIBRARY_PATH
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

echo "Config: ${CONFIG}"
echo "Data path override: ${DATA_PATH}"

python3 -m nbf.domains.lorenz96.generate_data \
    --config="${CONFIG}" \
    --set "data_path=${DATA_PATH}"
