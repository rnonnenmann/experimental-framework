from pathlib import Path
from datetime import datetime
import os

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.diffusion import DiffusionConfig
from lerobot.scripts.lerobot_train import train

import wandb

wandb.login()

workspace_root = Path(__file__).resolve().parents[2]
outputs_root = Path(os.environ.get("MASTERARBEIT_OUTPUTS", workspace_root / "outputs"))
dataset_root = Path(
    os.environ.get(
        "TRAINING_DATASET_ROOT",
        outputs_root / "policy_training_data/small_t_mouse/lerobot_dataset",
    )
)
pretrained_policy_path = Path(
    os.environ.get(
        "PRETRAINED_POLICY_PATH",
        workspace_root / "gym-pusht-main/policies/diffusion_pusht_migrated",
    )
)

# Use None for all episodes, or set a list of candidate episode indices.
selected_episodes = None

policy_cfg = DiffusionConfig.from_pretrained(pretrained_policy_path)
policy_cfg.pretrained_path = pretrained_policy_path
policy_cfg.device = "cuda"
policy_cfg.push_to_hub = False
policy_cfg.optimizer_lr = 1e-5

run_name = f"pusht_obstacle_finetuned_{datetime.now():%Y%m%d_%H%M%S}"
output_dir = Path(
    os.environ.get(
        "POLICY_OUTPUT_DIR",
        outputs_root / "policies",
    )
) / run_name

cfg = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id="local/obstacle_mouse_run_20261028_150040",
        root=str(dataset_root),
        episodes=selected_episodes,
        # eval_split=0.2,  # Hold out the last 20% of selected episodes.
    ),
    policy=policy_cfg,
    output_dir=output_dir,
    job_name=run_name,
    batch_size=64,
    steps=30000,
    save_freq=5000,
    env_eval_freq=0,
    # eval_steps=100,
    num_workers=0,
    log_freq=500,
    persistent_workers=True,
)

cfg.wandb.enable = True
train(cfg)
wandb.finish()
