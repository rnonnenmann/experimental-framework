#!/usr/bin/env bash
# Fine-tune the PushT diffusion policy on Karolina.
# Submit from the project checkout with:
#   sbatch lerobot-main/scripts/train_policy.sh
# Set LEROBOT_REPO, LEROBOT_VENV_ACTIVATE, TRAINING_DATASET_ROOT,
# PRETRAINED_POLICY_PATH, and POLICY_OUTPUT_DIR through the environment.

#SBATCH --job-name=pusht-obstacle-policy
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=pusht-obstacle-policy-%j.out
#SBATCH --error=pusht-obstacle-policy-%j.err

set -euo pipefail

# TorchCodec needs the matching FFmpeg shared libraries at runtime. This module
# is available on Karolina; adjust it if using a different software stack.
module load FFmpeg/7.1.2-GCCcore-14.3.0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${LEROBOT_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VENV_ACTIVATE="${LEROBOT_VENV_ACTIVATE:-$HOME/lerobot/bin/activate}"
TRAINING_DATASET_ROOT="${TRAINING_DATASET_ROOT:-${SCRATCH:-/scratch}/policy_training_data/pusht_obstacle/lerobot_dataset}"
PRETRAINED_POLICY_PATH="${PRETRAINED_POLICY_PATH:-${SCRATCH:-/scratch}/policies/diffusion_pusht_migrated}"
POLICY_OUTPUT_DIR="${POLICY_OUTPUT_DIR:-${SCRATCH:-/scratch}/policies/trained}"

if [[ ! -d "$REPO" ]]; then
  echo "LeRobot repository not found: $REPO" >&2
  exit 1
fi
if [[ ! -r "$VENV_ACTIVATE" ]]; then
  echo "Python environment activation script not found: $VENV_ACTIVATE" >&2
  exit 1
fi
if [[ ! -d "$TRAINING_DATASET_ROOT" ]]; then
  echo "Training dataset not found: $TRAINING_DATASET_ROOT" >&2
  echo "Set TRAINING_DATASET_ROOT to the directory containing the LeRobot dataset." >&2
  exit 1
fi
if [[ ! -d "$PRETRAINED_POLICY_PATH" ]]; then
  echo "Pretrained policy not found: $PRETRAINED_POLICY_PATH" >&2
  echo "Set PRETRAINED_POLICY_PATH to the migrated diffusion policy directory." >&2
  exit 1
fi

# Load W&B credentials on the compute node. Keep the API key out of this script
# and store it in ~/.wandb_env with mode 600.
if [[ ! -r "$HOME/.wandb_env" ]]; then
  echo "[finetune] Missing readable credentials file: $HOME/.wandb_env" >&2
  exit 1
fi
source "$HOME/.wandb_env"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.wandb_env}"
export WANDB_API_KEY

cd "$REPO"
source "$VENV_ACTIVATE"
mkdir -p "$POLICY_OUTPUT_DIR"

export TRAINING_DATASET_ROOT PRETRAINED_POLICY_PATH POLICY_OUTPUT_DIR
export WANDB_DIR="${WANDB_DIR:-${SCRATCH:-/scratch}/wandb}"
mkdir -p "$WANDB_DIR"

echo "[lerobot-policy] repo=$REPO"
echo "[lerobot-policy] node=${SLURMD_NODENAME:-unknown}"
echo "[lerobot-policy] python=$(command -v python)"
echo "[lerobot-policy] dataset=$TRAINING_DATASET_ROOT"
echo "[lerobot-policy] pretrained_policy=$PRETRAINED_POLICY_PATH"
echo "[lerobot-policy] output=$POLICY_OUTPUT_DIR"
echo "[lerobot-policy] ffmpeg=$(ffmpeg -version | sed -n '1p')"

exec python scripts/train_policy.py
