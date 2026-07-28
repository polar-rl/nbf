#!/bin/bash
#SBATCH --job-name=l96-train
#SBATCH --account=rrg-mbowling-ad
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#
# L96 NBF training on rorqual. Stages the corpus to node-local NVMe (PE Lustre
# lesson). Requeue-safe: run_training resumes from save_path/model.eqx (LR
# schedule restarts). Submit with a dependency on datagen:
#   sbatch --dependency=afterok:<datagen_jobid> --export=ALL,CONFIG=... rorqual_train_l96.sbatch
#
# Required env:
#   CONFIG   repo-relative config yaml (same one datagen ran)
# Optional env:
#   DATA_ROOT  default $SCRATCH/nbf/l96/<config-name>/data
#   SAVE_ROOT  default $SCRATCH/nbf/l96/<config-name>/results
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
SAVE_ROOT=${SAVE_ROOT:-$SCRATCH/nbf/l96/$NAME/results}
SRC_DIR="$DATA_ROOT/l96_belief_data_K20"

[ -f "$SRC_DIR/norm_stats.json" ] || { echo "FATAL: incomplete corpus at $SRC_DIR (no norm_stats.json)"; exit 1; }

mkdir -p "$SAVE_ROOT" "$SLURM_TMPDIR/data"
echo "Staging $SRC_DIR -> $SLURM_TMPDIR/data ..."
SECONDS=0
cp -r "$SRC_DIR" "$SLURM_TMPDIR/data/"
echo "Staged in ${SECONDS}s."

cd "$BASE/repos/nbf"
uv sync --frozen --extra gpu
PY=$SLURM_TMPDIR/venv/bin/python

for r in nbf embedex; do
  echo "$r $(git -C "$BASE/repos/$r" rev-parse HEAD)"
done > "$SAVE_ROOT/git-shas.txt"
echo "{\"job\": \"$SLURM_JOB_ID\", \"node\": \"$(hostname)\", \"name\": \"$NAME\"}" > "$SAVE_ROOT/meta.json"

echo "Config: $CONFIG  data: $SLURM_TMPDIR/data/  save: $SAVE_ROOT"
$PY -m nbf.domains.lorenz96.train \
    --config="$CONFIG" \
    --set "data_path=$SLURM_TMPDIR/data/" "save_path=$SAVE_ROOT/"

# Promote the checkpoint off scratch (60-day purge) to backed-up project space.
PROMOTE=$BASE/results/nbf-l96/$NAME
mkdir -p "$PROMOTE"
cp "$SAVE_ROOT"/model.eqx "$SAVE_ROOT"/model_config.json "$SAVE_ROOT"/data_config.json "$PROMOTE"/ 2>/dev/null || true
cp "$SAVE_ROOT"/loss-curve.pdf "$SAVE_ROOT"/git-shas.txt "$SAVE_ROOT"/meta.json "$PROMOTE"/ 2>/dev/null || true
echo "Promoted checkpoint to $PROMOTE"
echo TRAIN-OK
