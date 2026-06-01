#!/bin/bash
#SBATCH --job-name=pe-eval
#SBATCH --account=aip-nathanst
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gpus-per-task=1
#SBATCH --array=2,3,4
#SBATCH --requeue
#SBATCH --output=/home/solinas/scratch/logs/pe_eval_e%a_%j.out
#SBATCH --mail-user=solinas@ualberta.ca
#SBATCH --mail-type=FAIL,REQUEUE,END
#
# SLURM batch script for PE evaluation. Array task IDs select experiment groups:
#   2 -> E2 (cross-k MMD + per-policy at k=1,2,3,5)
#   3 -> E3 (OOD regimes, k=1)
#   4 -> E4 (observation-noise sigma sweep)
#
# Tunables (env vars):
#   PROJECT_DIR    repo checkout on the cluster
#   SCRATCH_BASE   scratch root containing results/
#   NUM_EPISODES, EPISODE_LENGTH, NUM_SEEDS
set -euo pipefail

EID=${SLURM_ARRAY_TASK_ID}
PROJECT_DIR=${PROJECT_DIR:-/home/solinas/projects/aip-nathanst/solinas/polar-rl/nbf}
SCRATCH_BASE=${SCRATCH_BASE:-/home/solinas/scratch/nbf/pe}

cd "${PROJECT_DIR}"

module load cuda
unset LD_LIBRARY_PATH
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

RESULTS="${SCRATCH_BASE}/results"
MODELS="1:${RESULTS}/k1/nbf,2:${RESULTS}/k2/nbf,3:${RESULTS}/k3/nbf,5:${RESULTS}/k5/nbf"

NUM_EPISODES=${NUM_EPISODES:-50}
EPISODE_LENGTH=${EPISODE_LENGTH:-15}
NUM_SEEDS=${NUM_SEEDS:-3}

run() {
    local exps="$1" model_dir="$2" out_dir="$3"; shift 3
    mkdir -p "${out_dir}"
    python3 -m nbf.domains.pursuit_evasion.evaluate \
        --experiments="${exps}" \
        --model_dir="${model_dir}" \
        --model_dirs_per_k="${MODELS}" \
        --output_dir="${out_dir}" \
        --num_episodes="${NUM_EPISODES}" \
        --episode_length="${EPISODE_LENGTH}" \
        --num_seeds="${NUM_SEEDS}" \
        "$@"
}

echo "EID=${EID}"
echo "Results base: ${RESULTS}"

case "${EID}" in
    2)
        run 2 "${RESULTS}/k1/nbf" "${RESULTS}/cross_k/eval" \
            --e2_ks=1,2,3,5
        ;;
    3)
        run 3 "${RESULTS}/k1/nbf" "${RESULTS}/k1/eval"
        ;;
    4)
        run 4 "${RESULTS}/k1/nbf" "${RESULTS}/k1/eval"
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID=${EID}" >&2
        exit 1
        ;;
esac
