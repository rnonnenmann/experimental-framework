#!/usr/bin/env bash
# Fine-tune NanoWM-B/2 on PushT with the settings from notebooks/train_wm.ipynb.
# Add the account option required by your cluster when submitting, for example:
#   sbatch --account=<cluster-account> scripts/finetune_wm.sh

#SBATCH --job-name=wm-finetune-large
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=wm-finetune-%j.out
#SBATCH --error=wm-finetune-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -x "$SUBMIT_DIR/nanowm" ]]; then
  REPO="$(cd -- "$SUBMIT_DIR" && pwd -P)"
elif [[ -x "$SUBMIT_DIR/../nanowm" ]]; then
  REPO="$(cd -- "$SUBMIT_DIR/.." && pwd -P)"
else
  echo "[finetune] Cannot find nanowm launcher from submit directory: $SUBMIT_DIR" >&2
  echo "[finetune] Submit from the nano-world-model-main repository root or its scripts/ directory." >&2
  exit 1
fi
cd "$REPO"

PROJECT_SCRATCH="${PROJECT_SCRATCH:-${SCRATCH:-/scratch}}"
export RESULTS_DIR="${RESULTS_DIR:-$PROJECT_SCRATCH/results}"
mkdir -p "$RESULTS_DIR"

export PUSHT_TRAIN_DIR="${PUSHT_TRAIN_DIR:-$PROJECT_SCRATCH/data/pushT_large/train}"
export PUSHT_VAL_DIR="${PUSHT_VAL_DIR:-$PROJECT_SCRATCH/data/pushT_large/val}"
export ORIGINAL_PUSHT_TRAIN_DIR="${ORIGINAL_PUSHT_TRAIN_DIR:-$PROJECT_SCRATCH/data/pushT_original/train}"
export ORIGINAL_PUSHT_VAL_DIR="${ORIGINAL_PUSHT_VAL_DIR:-$PROJECT_SCRATCH/data/pushT_original/val}"
export FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT:-$PROJECT_SCRATCH/checkpoint/model.safetensors}"
export REPLAY_RATIO="${REPLAY_RATIO:-0.5}"
export REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-1.0}"

# Load W&B credentials on the compute node. Keep the API key out of this script
# and store it in ~/.wandb_env with mode 600.
if [[ ! -r "$HOME/.wandb_env" ]]; then
  echo "[finetune] Missing readable credentials file: $HOME/.wandb_env" >&2
  exit 1
fi
source "$HOME/.wandb_env"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in ~/.wandb_env}"
export WANDB_API_KEY

echo "[finetune] repo=$REPO"
echo "[finetune] node=${SLURMD_NODENAME:-unknown}"
echo "[finetune] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[finetune] train_data=$PUSHT_TRAIN_DIR"
echo "[finetune] val_data=$PUSHT_VAL_DIR"
echo "[finetune] pretrained=$FINETUNE_CHECKPOINT"
echo "[finetune] replay_ratio=$REPLAY_RATIO"
echo "[finetune] replay_buffer_size=$REPLAY_BUFFER_SIZE"
echo "[finetune] results=$RESULTS_DIR"

exec ./nanowm python src/main.py \
  dataset=dino_wm/pusht \
  experiment=dino_wm_pusht \
  model=nanowm_b2 \
  "dataset.loader.data_path_train=$PUSHT_TRAIN_DIR" \
  "dataset.loader.data_path_val=$PUSHT_VAL_DIR" \
  "+dataset.loader.original_data_path_train=$ORIGINAL_PUSHT_TRAIN_DIR"\
  "+dataset.loader.original_data_path_val=$ORIGINAL_PUSHT_VAL_DIR"\
  +dataset.loader.use_ER=false \
  dataset.loader.replay_ratio="$REPLAY_RATIO" \
  dataset.loader.replay_buffer_size="$REPLAY_BUFFER_SIZE" \
  dataset.loader.use_relative_actions=true \
  dataset.loader.split_ratio=1.0 \
  dataset.loader.train_slice_mode=exhaustive \
  dataset.loader.val_slice_mode=exhaustive \
  dataset.loader.action_scale=100.0 \
  dataset.loader.with_velocity=false \
  "experiment.pretrained=$FINETUNE_CHECKPOINT" \
  experiment.training.max_steps=20000 \
  experiment.training.checkpointing.across_timesteps.every_n_train_steps=5000 \
  experiment.training.optimizer.lr_warmup_steps=250 \
  experiment.training.optimizer.lr=1e-5 \
  experiment.training.optimizer.lr_scheduler=cosine \
  experiment.training.batch_size=16 \
  experiment.training.gradient_accumulation=1 \
  experiment.infra.mixed_precision=true \
  experiment.infra.compile=true \
  wandb.enabled=true
