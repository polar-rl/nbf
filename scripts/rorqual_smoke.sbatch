#!/bin/bash
#SBATCH --job-name=l96-smoke
#SBATCH --account=rrg-mbowling-ad
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=8000M
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#
# Rorqual smoke: sbatch is UNVERIFIED on rorqual (salloc only, per
# CLUSTER_SETUP.md). Run this once before any long job. Submit from
# ~/projects/rrg-mbowling-ad/solinas/slurm so the log lands there.
set -euo pipefail
: "${SLURM_TMPDIR:?SLURM_TMPDIR unset}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

BASE=$HOME/projects/rrg-mbowling-ad/solinas
export UV_CACHE_DIR=$BASE/uv-cache
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT=$SLURM_TMPDIR/venv

cd "$BASE/repos/nbf"
uv sync --frozen --extra gpu
PY=$SLURM_TMPDIR/venv/bin/python

$PY -c "import jax; print('devices:', jax.devices())"
$PY -c "import nbf.domains.lorenz96.run_grid; print('imports OK')"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo SMOKE-OK
