import gymnasium as gym
import ale_py
import torch
import numpy as np
from pathlib import Path

from dqn_b import DQN, stack_frames

model_path = Path.cwd() / 'models' / 'best_dqn_pong.pth'
model_path = model_path.resolve()
num_episodes = 5

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('ALE/Pong-v5', render_mode=None)
n_actions = env.action_space.n

policy_net = DQN((4, 84, 84), n_actions).to(device)
checkpoint = torch.load(model_path, map_location=device)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
policy_net.eval()

episode_rewards = []
for episode in range(num_episodes):
    obs, _ = env.reset()
    stacked_frames = None
    state, stacked_frames = stack_frames(stacked_frames, obs, is_new_episode=True)
    
    episode_reward = 0
    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state, stacked_frames = stack_frames(stacked_frames, next_obs, is_new_episode=False)
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    episode_rewards.append(episode_reward)
    print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

env.close()
avg_reward = np.mean(episode_rewards)
print(f"Average test reward: {avg_reward:.2f}")
