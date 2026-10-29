from datetime import datetime
import os
from pathlib import Path
from time import time

import gymnasium as gym
import gym_pusht
import numpy as np
import pygame
import imageio.v2 as imageio
from pathlib import Path
import torch
from huggingface_hub import snapshot_download

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from safetensors import safe_open

# Download PushT Policy
policy_path = Path(os.environ.get("POLICY_PATH", Path(__file__).resolve().parents[1] / "policies"))
policy_path.mkdir(parents=True, exist_ok=True)
POLICY_ID = "RNonnenmann/large_pusht_diffusion"
POLICY_REVISION = "bc6f5cd4dcab061c6f4450f145787a69d0668f4e"
diffusion_pusht = snapshot_download(POLICY_ID, revision=POLICY_REVISION,
                        allow_patterns=["config.json", "model.safetensors", "policy_preprocessor_step_3_normalizer_processor.safetensors"],
                        local_dir=str(policy_path / "small_pusht_diffusion"))

# Load the checkpoint for inference
device = "cuda"
policy = DiffusionPolicy.from_pretrained(diffusion_pusht, map_location=device)
policy.to(device)
policy.eval()

# The checkpoint stores its input and output normalization statistics in the safetensors file
with safe_open(str(Path(diffusion_pusht) / "policy_preprocessor_step_3_normalizer_processor.safetensors"), framework="pt", device="cpu") as f:
    image_mean = f.get_tensor("observation.image.mean").to(device).view(1, 3, 1, 1)
    image_std = f.get_tensor("observation.image.std").to(device).view(1, 3, 1, 1)
    state_min = f.get_tensor("observation.state.min").to(device)
    state_max = f.get_tensor("observation.state.max").to(device)
    action_min = f.get_tensor("action.min").to(device)
    action_max = f.get_tensor("action.max").to(device)


# Change the number of diffusion steps
policy.diffusion.noise_scheduler = DDIMScheduler.from_config(
    policy.diffusion.noise_scheduler.config
)
policy.diffusion.num_inference_steps = 10


# PushT advances by one action every 0.1 simulated seconds.
recording_fps = 10
steps = 150
episodes = 50
scale = 40

# Define output directory
data_path = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parents[1] / "outputs" / "wm_training_data"))
run_path = data_path / datetime.now().strftime("run_%Y%m%d_%H%M%S")
video_path = run_path / "obses"
video_path.mkdir(parents=True, exist_ok=True)

env = gym.make(
    "gym_pusht/PushT-v0",
    render_mode="rgb_array",
    obs_type="pixels_agent_pos",
    scale=scale,
    max_episode_steps=300,
)
base_env = env.unwrapped

abs_actions = np.zeros((episodes, steps, 2), dtype=np.float32)
rel_actions = np.zeros((episodes, steps, 2), dtype=np.float32)
agent_positions = np.zeros((episodes, steps, 2), dtype=np.float32)
full_states = np.zeros((episodes, steps, 5), dtype=np.float32)
rewards = np.zeros((episodes, steps), dtype=np.float32)
coverages = np.zeros((episodes, steps), dtype=np.float32)
successes = np.zeros((episodes, steps), dtype=np.bool_)
lengths = np.zeros(episodes, dtype=np.int64)
loop_clock = pygame.time.Clock()

for episode in range(episodes):
    print(f"Starting episode {episode+1}/{episodes}")
    start_time = time()
    observation, info = env.reset()
    episode_video_path = video_path / f"episode_{episode:03d}.mp4"

    with imageio.get_writer(
        episode_video_path,
        fps=recording_fps,
        codec="libx264",
        is_batch=False,
    ) as video:
        for step in range(steps):

            # The policy expects an RGB image and the current agent position.
            image = torch.as_tensor(observation["pixels"], dtype=torch.float32, device=device)
            image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
            image = (image - image_mean) / image_std
            agent_pos = torch.as_tensor(
                observation["agent_pos"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            agent_pos = 2 * (agent_pos - state_min) / (state_max - state_min) - 1
            policy_observation = {
                "observation.image": image,
                "observation.state": agent_pos,
            }

            with torch.inference_mode():
                action = policy.select_action(policy_observation)

            # `select_action` returns normalized actions; map them back to absolute PushT coordinates.
            abs_action = ((action.squeeze(0) + 1) / 2 * (action_max - action_min) + action_min)
            abs_action = abs_action.cpu().numpy().astype(np.float32)
            abs_action = np.clip(abs_action, env.action_space.low, env.action_space.high)


            agent_pos = np.asarray(observation["agent_pos"], dtype=np.float32)
            full_state = np.array(
                [
                    *base_env.agent.position,
                    *base_env.block.position,
                    base_env.block.angle % (2 * np.pi),
                ],
                dtype=np.float32,
            )

            # Capture the scene drawn by the environment.
            frame = base_env._get_img(
                base_env._draw(), width=224, height=224, render_action=False
            )
            video.append_data(frame)

            rel_action = abs_action - agent_pos

            # Save the current action and state
            abs_actions[episode, step] = abs_action
            rel_actions[episode, step] = rel_action
            agent_positions[episode, step] = agent_pos
            full_states[episode, step] = full_state

            observation, reward, terminated, truncated, info = env.step(abs_action)
            rewards[episode, step] = reward
            coverages[episode, step] = info["coverage"]
            successes[episode, step] = info["is_success"]
            lengths[episode] = step + 1

            if terminated or truncated:
                break

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Finished episode {episode+1} in {elapsed_time:.2f} seconds")

env.close()

# Save data tensors
torch.save(torch.from_numpy(abs_actions), run_path / "abs_actions.pth")
torch.save(torch.from_numpy(rel_actions), run_path / "rel_actions.pth")
torch.save(torch.from_numpy(full_states), run_path / "states.pth")
torch.save(torch.from_numpy(lengths), run_path / "seq_lengths.pth")
