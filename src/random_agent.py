import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/Pong-v5', render_mode="human")

observation, info = env.reset()
episode_over = False

while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
