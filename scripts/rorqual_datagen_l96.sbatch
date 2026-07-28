#!/bin/bash
#SBATCH --job-name=l96-datagen
#SBATCH --account=rrg-mbowling-ad
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#
# L96 gold-SMC datagen on rorqual. Walltime: 12h default fits hard_4m with
# margin; the easy config is far lighter — submit it with --time=3:00:00 to
# drop into the 3h scheduling bucket (more eligible nodes + backfill).
# Requeue-safe: generate_data skips only when
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

# BLOCK_SIZE: scenarios per vmapped gold-SMC block. Must divide num_scenarios.
# H100-80GB guidance: easy (2M particles) -> 20; hard_4m (4M) is already 16 in
# its config. Unset = use the config value.
#
# Generate onto node-local NVMe, then one sequential copy to scratch at the
# end (Lustre writeback throttling turned incremental memmap writes into the
# bottleneck: 4s/block decayed to 23s/block on the first run). The corpus dir
# is copied under a temp name and atomically renamed, so the norm_stats.json
# completion guard can never see a half-copied corpus.
LOCAL_OUT=$SLURM_TMPDIR/datagen_out
mkdir -p "$LOCAL_OUT"

OVERRIDES=("data_path=$LOCAL_OUT/")
if [ -n "${BLOCK_SIZE:-}" ]; then
  OVERRIDES+=("data.block_size=$BLOCK_SIZE")
fi

echo "Config: $CONFIG -> $LOCAL_OUT (overrides: ${OVERRIDES[*]})"
$PY -m nbf.domains.lorenz96.generate_data \
    --config="$CONFIG" \
    --set "${OVERRIDES[@]}"

CORPUS_DIR=$(basename "$(find "$LOCAL_OUT" -maxdepth 1 -mindepth 1 -type d | head -1)")
[ -f "$LOCAL_OUT/$CORPUS_DIR/norm_stats.json" ] || { echo "FATAL: generation incomplete"; exit 1; }
echo "Copying $LOCAL_OUT/$CORPUS_DIR -> $DATA_ROOT/$CORPUS_DIR ..."
SECONDS=0
rm -rf "$DATA_ROOT/$CORPUS_DIR.tmp"
cp -r "$LOCAL_OUT/$CORPUS_DIR" "$DATA_ROOT/$CORPUS_DIR.tmp"
rm -rf "$DATA_ROOT/$CORPUS_DIR"
mv "$DATA_ROOT/$CORPUS_DIR.tmp" "$DATA_ROOT/$CORPUS_DIR"
echo "Copied in ${SECONDS}s."
echo DATAGEN-OK
