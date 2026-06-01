#!/bin/bash
#SBATCH --job-name=pe-datagen
#SBATCH --account=aip-nathanst
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-task=1
#SBATCH --array=1,2,3,5
#SBATCH --requeue
#SBATCH --output=/home/solinas/scratch/logs/pe_datagen_k%a_%j.out
#SBATCH --mail-user=solinas@ualberta.ca
#SBATCH --mail-type=FAIL,REQUEUE,END
#
# SLURM batch script for PE belief-data generation. Array task index = k.
#
# Tunables (env vars):
#   PROJECT_DIR    repo checkout on the cluster
#   SCRATCH_BASE   scratch root for data + logs
set -euo pipefail

K=${SLURM_ARRAY_TASK_ID}
PROJECT_DIR=${PROJECT_DIR:-/home/solinas/projects/aip-nathanst/solinas/polar-rl/nbf}
SCRATCH_BASE=${SCRATCH_BASE:-/home/solinas/scratch/nbf/pe}

CONFIG="nbf/domains/pursuit_evasion/configs/nbf_k${K}.yaml"
DATA_PATH="${SCRATCH_BASE}/data/"

mkdir -p "${DATA_PATH}" "/home/solinas/scratch/logs"

cd "${PROJECT_DIR}"

module load cuda
unset LD_LIBRARY_PATH
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

echo "K=${K}"
echo "Config: ${CONFIG}"
echo "Data path override: ${DATA_PATH}"

python3 -m nbf.domains.pursuit_evasion.generate_data \
    --config="${CONFIG}" \
    --set "data_path=${DATA_PATH}"
