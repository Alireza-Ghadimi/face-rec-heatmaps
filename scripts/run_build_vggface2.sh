#!/bin/bash
#SBATCH --job-name=build-vggface2
#SBATCH --account=aip-jelder
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail

# Activate your environment (adjust path/name as needed)
source ~/projects/aip-jelder/ghadimi/face-rec-heatmaps/.venv/bin/activate  # or: module load python && source ~/project/venv/bin/activate

# Use local scratch for HF caches to avoid quota issues
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="${SLURM_TMPDIR:-/tmp}/hf_datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

cd "$SLURM_SUBMIT_DIR"

# python scripts/build_vggface2.py \
#   --dataset anhnct/vggface2 \
#   --split train \
#   --out_dir data_vggface2 \
#   --streaming \
#   --sample_prob 0.2 \
#   --max_samples 5000

python scripts/build_vggface2.py \
  --dataset logasja/vggface2 \
  --split train \
  --out_dir data_vggface2 \
  --chunk_size 1000 \
  --chunk_dir data_vggface2/chunks \
  --csv_path data_vggface2_first_row.csv \
  --save_images False \
  --streaming 