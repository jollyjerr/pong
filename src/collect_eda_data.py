import h5py
import gymnasium as gym
import ale_py
import os
import numpy as np

from tqdm import tqdm

EPISODES = 5
FRAMES_PER_EPISODE = 1000

gym.register_envs(ale_py)

env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
observation, info = env.reset()

file_path = os.path.join(os.getcwd(), 'data/eda.h5')
print(f"Creating dataset at: {file_path}")

with h5py.File(file_path, 'w') as f:
    observations_dataset = f.create_dataset('observations', (0, 210, 160, 3), maxshape=(
        None, 210, 160, 3), dtype=np.uint8, compression="gzip")
    actions_dataset = f.create_dataset('actions', (0,), maxshape=(None,), dtype=np.int32)
    rewards_dataset = f.create_dataset('rewards', (0,), maxshape=(None,), dtype=np.float32)
    terminated_flags_dataset = f.create_dataset('terminated', (0,), maxshape=(None,), dtype=bool)
    truncated_flags_dataset = f.create_dataset('truncated', (0,), maxshape=(None,), dtype=bool)
    episode_boundaries = f.create_dataset(
        'episode_boundaries', (0, 2), maxshape=(None, 2), dtype=int)

    current_frame_idx = 0
    for i_episode in tqdm(range(EPISODES)):
        observation, info = env.reset()
        episode_start_idx = current_frame_idx
        episode_over = False
        frame_count = 0

        while not episode_over and frame_count < FRAMES_PER_EPISODE:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)

            observations_dataset.resize(current_frame_idx + 1, axis=0)
            actions_dataset.resize(current_frame_idx + 1, axis=0)
            rewards_dataset.resize(current_frame_idx + 1, axis=0)
            terminated_flags_dataset.resize(current_frame_idx + 1, axis=0)
            truncated_flags_dataset.resize(current_frame_idx + 1, axis=0)

            observations_dataset[current_frame_idx] = observation
            actions_dataset[current_frame_idx] = action
            rewards_dataset[current_frame_idx] = reward
            terminated_flags_dataset[current_frame_idx] = terminated
            truncated_flags_dataset[current_frame_idx] = truncated

            observation = next_observation
            episode_over = terminated or truncated
            current_frame_idx += 1
            frame_count += 1

        episode_boundaries.resize(i_episode + 1, axis=0)
        episode_boundaries[i_episode] = [episode_start_idx, current_frame_idx - 1]

env.close()
print("Done.")
