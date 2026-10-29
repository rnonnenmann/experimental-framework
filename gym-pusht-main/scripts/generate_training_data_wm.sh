#!/usr/bin/env bash
# Generate PushT demonstrations for the world model on Karolina.
# Submit from the project checkout with:
#   sbatch --account=<cluster-account> gym-pusht-main/scripts/generate_training_data_wm.sh
# Set LEROBOT_VENV_ACTIVATE if the environment is not at "$HOME/lerobot/bin/activate".

#SBATCH --job-name=pusht-wm-data
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=pusht-wm-data-%j.out
#SBATCH --error=pusht-wm-data-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${GYM_PUSHT_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VENV="${LEROBOT_VENV_ACTIVATE:-$HOME/lerobot/bin/activate}"

if [[ ! -d "$REPO" ]]; then
  echo "PushT repository not found: $REPO" >&2
  exit 1
fi
if [[ ! -r "$VENV" ]]; then
  echo "Python environment activation script not found: $VENV" >&2
  exit 1
fi

# Read the token from a private file
export HF_TOKEN="$(cat ~/.hf_token)"

cd "$REPO"
source "$VENV"

# Keep generated data and the downloaded policy on project scratch storage.
export POLICY_PATH="${POLICY_PATH:-${SCRATCH:-/scratch}/policies}"
export DATA_PATH="${DATA_PATH:-${SCRATCH:-/scratch}/wm_training_data}"
mkdir -p "$POLICY_PATH" "$DATA_PATH"

# Pygame only renders offscreen frames in this batch job.
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

echo "[pusht-wm-data] repo=$REPO"
echo "[pusht-wm-data] node=${SLURMD_NODENAME:-unknown}"
echo "[pusht-wm-data] python=$(command -v python)"
echo "[pusht-wm-data] policy_path=$POLICY_PATH"
echo "[pusht-wm-data] data_path=$DATA_PATH"

exec python scripts/generate_training_data_wm.py
