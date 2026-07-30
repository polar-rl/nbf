#!/bin/bash
#SBATCH --job-name=l96-train
#SBATCH --account=aip-nathanst
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-task=1
#SBATCH --requeue
#SBATCH --output=/home/solinas/scratch/logs/l96_train_%j.out
#SBATCH --mail-user=solinas@ualberta.ca
#SBATCH --mail-type=FAIL,REQUEUE,END
#
# SLURM batch script for L96 NBF training (K=20).
#
# Tunables (env vars):
#   PROJECT_DIR    repo checkout on the cluster
#   SCRATCH_BASE   scratch root for data, models, logs
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/solinas/projects/aip-nathanst/solinas/polar-rl/nbf}
SCRATCH_BASE=${SCRATCH_BASE:-/home/solinas/scratch/nbf/l96}

CONFIG="nbf/domains/lorenz96/configs/nbf_l96.yaml"
SCRATCH_DATA_PATH="${SCRATCH_BASE}/data/"
SAVE_PATH="${SCRATCH_BASE}/results/k20/nbf/"

# Stage data to node-local NVMe ($SLURM_TMPDIR) for fast random reads (PE Lustre
# lesson: networked /scratch is too slow for random snapshot+row picks).
LOCAL_DATA_PATH="${SLURM_TMPDIR}/data/"
SRC_DATA_DIR="${SCRATCH_DATA_PATH}/l96_belief_data_K20"
DST_DATA_DIR="${LOCAL_DATA_PATH}/l96_belief_data_K20"

mkdir -p "${SCRATCH_DATA_PATH}" "${SAVE_PATH}" "${LOCAL_DATA_PATH}" "/home/solinas/scratch/logs"

if [ -d "${SRC_DATA_DIR}" ]; then
    echo "Staging ${SRC_DATA_DIR} -> ${DST_DATA_DIR} ..."
    SECONDS=0
    cp -r "${SRC_DATA_DIR}" "${DST_DATA_DIR}"
    echo "Staging done in ${SECONDS}s."
    DATA_PATH="${LOCAL_DATA_PATH}"
else
    echo "No cached data at ${SRC_DATA_DIR}; will generate to scratch."
    DATA_PATH="${SCRATCH_DATA_PATH}"
fi

cd "${PROJECT_DIR}"

module load cuda
unset LD_LIBRARY_PATH
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

echo "Config: ${CONFIG}"
echo "Data path override: ${DATA_PATH}"
echo "Save path override: ${SAVE_PATH}"

python3 -m nbf.domains.lorenz96.train \
    --config="${CONFIG}" \
    --set "data_path=${DATA_PATH}" "save_path=${SAVE_PATH}"
