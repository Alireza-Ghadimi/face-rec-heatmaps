#!/bin/bash
#SBATCH --job-name=ae-train
#SBATCH --account=aip-jelder
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

# Activate your environment (adjust path)
source ~/projects/aip-jelder/ghadimi/face-rec-heatmaps/.venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

python -m src.autoencoder_train \
  --chunk_dir data_vggface2/chunks \
  --device cuda \
  --epochs 1000 \
  --batch_size 128 \
  --lr 1e-3 \
  --save_path autoencoder.pt \
  --device cuda
