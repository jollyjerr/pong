import h5py
import gymnasium as gym
import ale_py
import os

NUMBER_OF_EPISODES = 10
NUMBER_OF_FRAMES = 2000

gym.register_envs(ale_py)

env = gym.make('ALE/Pong-v5', render_mode="rgb_array")
observation, info = env.reset()

file_path = os.path.join(os.getcwd(), 'data/eda.h5')
print(f"creating dataset for EDA at: {file_path}")

with h5py.File(file_path, 'w') as f:
    pass
