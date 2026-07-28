#!/bin/bash
#SBATCH --job-name=l96-datagen
#SBATCH --account=rrg-mbowling-ad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#
# L96 gold-SMC datagen on rorqual. Requeue-safe: generate_data skips only when
# norm_stats.json (written last) exists, so a preempted run regenerates.
#
# Required env:
#   CONFIG   repo-relative config, e.g. nbf/domains/lorenz96/configs/nbf_l96_easy.yaml
# Optional env:
#   DATA_ROOT  default $SCRATCH/nbf/l96/<config-name>/data
set -euo pipefail
: "${SLURM_TMPDIR:?SLURM_TMPDIR unset}"
: "${CONFIG:?set CONFIG to a repo-relative yaml path}"
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
SCRATCH=${SCRATCH:-/scratch/$USER}

BASE=$HOME/projects/rrg-mbowling-ad/solinas
export UV_CACHE_DIR=$BASE/uv-cache
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT=$SLURM_TMPDIR/venv

NAME=$(basename "$CONFIG" .yaml)
DATA_ROOT=${DATA_ROOT:-$SCRATCH/nbf/l96/$NAME/data}
mkdir -p "$DATA_ROOT"

cd "$BASE/repos/nbf"
uv sync --frozen --extra gpu
PY=$SLURM_TMPDIR/venv/bin/python

for r in nbf embedex; do
  echo "$r $(git -C "$BASE/repos/$r" rev-parse HEAD)"
done > "$DATA_ROOT/git-shas.txt"
cp "$CONFIG" "$DATA_ROOT/config.yaml"
echo "{\"job\": \"$SLURM_JOB_ID\", \"node\": \"$(hostname)\", \"name\": \"$NAME\"}" > "$DATA_ROOT/meta.json"

echo "Config: $CONFIG -> $DATA_ROOT"
$PY -m nbf.domains.lorenz96.generate_data \
    --config="$CONFIG" \
    --set "data_path=$DATA_ROOT/"
echo DATAGEN-OK
