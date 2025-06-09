import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import ale_py
import cv2
import pandas as pd
import re

gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5')

def eda_dataset_highlights(file_path):
    with h5py.File(file_path, 'r') as f:
        total_frames = f['observations'].shape[0]
        episode_boundaries = f['episode_boundaries'][:]
        rewards_data = f['rewards'][:]
        actions = f['actions'][:]
        num_episodes = episode_boundaries.shape[0]

        episode_lengths = []
        for start_idx, end_idx in episode_boundaries:
            episode_lengths.append(end_idx - start_idx + 1)

        cumulative_rewards_per_episode = []
        for i, (start_idx, end_idx) in enumerate(episode_boundaries):
            episode_rewards = rewards_data[start_idx : end_idx + 1]
            cumulative_reward = np.sum(episode_rewards)
            cumulative_rewards_per_episode.append(cumulative_reward)
    
    action_names = [f"Action {i}" for i in range(env.action_space.n)]
    unique, counts = np.unique(actions, return_counts=True)
    action_freq = dict(zip(unique, counts))
    full_freq = {i: action_freq.get(i, 0) for i in range(env.action_space.n)}

    print(f"\nTotal number of frames collected: {total_frames}")
    print(f"Total number of episodes collected: {num_episodes}")
    print(f"\nMinimum episode length: {np.min(episode_lengths)}")
    print(f"Maximum episode length: {np.max(episode_lengths)}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
    print(f"Median episode length: {np.median(episode_lengths)}")

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_rewards_per_episode, label='Cumulative Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(full_freq.keys()), y=list(full_freq.values()))
    plt.xticks(ticks=range(env.action_space.n), labels=action_names, rotation=45)
    plt.title("Action Frequency Distribution of Random Agent", fontsize=16)
    plt.xlabel("Actions", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def eda_frame_highlights(file_path):
    with h5py.File(file_path, 'r') as f:
        sequence_start_idx = 100
        num_frames_to_show = 5
        obs = f['observations'][40]
        frames = f['observations'][sequence_start_idx:sequence_start_idx + num_frames_to_show]
        diff_images = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i].astype(np.float32) - frames[i-1].astype(np.float32))
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            diff_images.append(diff)

    fig2, axes2 = plt.subplots(1, num_frames_to_show - 1, figsize=(15, 3))
    for i in range(len(diff_images)):
        axes2[i].imshow(diff_images[i])
        axes2[i].set_title(f"Diff: Frame {i+1} - {i}")
        axes2[i].axis('off')
    plt.suptitle("Absolute Difference Between Consecutive Frames", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.show()

    gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) 
    resized_obs = cv2.resize(gray_obs, (84, 84), interpolation=cv2.INTER_LINEAR)
    small_resized_obs = cv2.resize(gray_obs, (64, 64), interpolation=cv2.INTER_LINEAR)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(obs)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(gray_obs, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(resized_obs, cmap='gray')
    plt.title("Resized (84x84)")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(small_resized_obs, cmap='gray')
    plt.title("Resized (64x64)")
    plt.axis('off')
    plt.show()

def training_sample(file_path):
    df = pd.read_csv(file_path)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(df['frame'], df['reward'], color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(df['frame'], df['epsilon'], color=color, linestyle='--', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Progress: Reward and Epsilon over Time')
    fig.tight_layout()
    plt.grid(True)
    plt.show()

log_data = [
                "Episode 0\n",
            "  Average Reward (last 100): -20.00\n",
            "  Current Epsilon: 0.999\n",
            "  Average Loss: 0.0000\n",
            "  Frame Count: 966\n",
            "  Elapsed Time: 0.1 min\n",
            "  Replay Buffer Size: 966\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -20.00\n",
            "Updated target network at frame 10000\n",
            "Updated target network at frame 20000\n",
            "Updated target network at frame 30000\n",
            "Updated target network at frame 40000\n",
            "Updated target network at frame 50000\n",
            "Updated target network at frame 60000\n",
            "Updated target network at frame 70000\n",
            "Updated target network at frame 80000\n",
            "Updated target network at frame 90000\n",
            "Episode 100\n",
            "  Average Reward (last 100): -20.33\n",
            "  Current Epsilon: 0.912\n",
            "  Average Loss: 0.0032\n",
            "  Frame Count: 93358\n",
            "  Elapsed Time: 4.8 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "Updated target network at frame 100000\n",
            "Updated target network at frame 110000\n",
            "Updated target network at frame 120000\n",
            "Updated target network at frame 130000\n",
            "Updated target network at frame 140000\n",
            "Updated target network at frame 150000\n",
            "Updated target network at frame 160000\n",
            "Updated target network at frame 170000\n",
            "Updated target network at frame 180000\n",
            "Episode 200\n",
            "  Average Reward (last 100): -20.22\n",
            "  Current Epsilon: 0.829\n",
            "  Average Loss: 0.0026\n",
            "  Frame Count: 189276\n",
            "  Elapsed Time: 10.0 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "Updated target network at frame 190000\n",
            "Updated target network at frame 200000\n",
            "Updated target network at frame 210000\n",
            "Updated target network at frame 220000\n",
            "Updated target network at frame 230000\n",
            "Updated target network at frame 240000\n",
            "Updated target network at frame 250000\n",
            "Updated target network at frame 260000\n",
            "Updated target network at frame 270000\n",
            "Updated target network at frame 280000\n",
            "Updated target network at frame 290000\n",
            "Episode 300\n",
            "  Average Reward (last 100): -19.96\n",
            "  Current Epsilon: 0.747\n",
            "  Average Loss: 0.0049\n",
            "  Frame Count: 294852\n",
            "  Elapsed Time: 15.6 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -19.96\n",
            "Updated target network at frame 300000\n",
            "Updated target network at frame 310000\n",
            "Updated target network at frame 320000\n",
            "Updated target network at frame 330000\n",
            "Updated target network at frame 340000\n",
            "Updated target network at frame 350000\n",
            "Updated target network at frame 360000\n",
            "Updated target network at frame 370000\n",
            "Updated target network at frame 380000\n",
            "Updated target network at frame 390000\n",
            "Updated target network at frame 400000\n",
            "Updated target network at frame 410000\n",
            "Episode 400\n",
            "  Average Reward (last 100): -19.35\n",
            "  Current Epsilon: 0.662\n",
            "  Average Loss: 0.0037\n",
            "  Frame Count: 417246\n",
            "  Elapsed Time: 22.1 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -19.35\n",
            "Updated target network at frame 420000\n",
            "Updated target network at frame 430000\n",
            "Updated target network at frame 440000\n",
            "Updated target network at frame 450000\n",
            "Updated target network at frame 460000\n",
            "Updated target network at frame 470000\n",
            "Updated target network at frame 480000\n",
            "Updated target network at frame 490000\n",
            "Updated target network at frame 500000\n",
            "Updated target network at frame 510000\n",
            "Updated target network at frame 520000\n",
            "Updated target network at frame 530000\n",
            "Updated target network at frame 540000\n",
            "Updated target network at frame 550000\n",
            "Episode 500\n",
            "  Average Reward (last 100): -18.58\n",
            "  Current Epsilon: 0.578\n",
            "  Average Loss: 0.0040\n",
            "  Frame Count: 555302\n",
            "  Elapsed Time: 29.4 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -18.58\n",
            "Updated target network at frame 560000\n",
            "Updated target network at frame 570000\n",
            "Updated target network at frame 580000\n",
            "Updated target network at frame 590000\n",
            "Updated target network at frame 600000\n",
            "Updated target network at frame 610000\n",
            "Updated target network at frame 620000\n",
            "Updated target network at frame 630000\n",
            "Updated target network at frame 640000\n",
            "Updated target network at frame 650000\n",
            "Updated target network at frame 660000\n",
            "Updated target network at frame 670000\n",
            "Updated target network at frame 680000\n",
            "Updated target network at frame 690000\n",
            "Updated target network at frame 700000\n",
            "Episode 600\n",
            "  Average Reward (last 100): -18.41\n",
            "  Current Epsilon: 0.498\n",
            "  Average Loss: 0.0036\n",
            "  Frame Count: 706662\n",
            "  Elapsed Time: 37.3 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -18.41\n",
            "Updated target network at frame 710000\n",
            "Updated target network at frame 720000\n",
            "Updated target network at frame 730000\n",
            "Updated target network at frame 740000\n",
            "Updated target network at frame 750000\n",
            "Updated target network at frame 760000\n",
            "Updated target network at frame 770000\n",
            "Updated target network at frame 780000\n",
            "Updated target network at frame 790000\n",
            "Updated target network at frame 800000\n",
            "Updated target network at frame 810000\n",
            "Updated target network at frame 820000\n",
            "Updated target network at frame 830000\n",
            "Updated target network at frame 840000\n",
            "Updated target network at frame 850000\n",
            "Updated target network at frame 860000\n",
            "Updated target network at frame 870000\n",
            "Updated target network at frame 880000\n",
            "Episode 700\n",
            "  Average Reward (last 100): -17.40\n",
            "  Current Epsilon: 0.420\n",
            "  Average Loss: 0.0041\n",
            "  Frame Count: 881190\n",
            "  Elapsed Time: 46.4 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -17.40\n",
            "Updated target network at frame 890000\n",
            "Updated target network at frame 900000\n",
            "Updated target network at frame 910000\n",
            "Updated target network at frame 920000\n",
            "Updated target network at frame 930000\n",
            "Updated target network at frame 940000\n",
            "Updated target network at frame 950000\n",
            "Updated target network at frame 960000\n",
            "Updated target network at frame 970000\n",
            "Updated target network at frame 980000\n",
            "Updated target network at frame 990000\n",
            "Updated target network at frame 1000000\n",
            "Updated target network at frame 1010000\n",
            "Updated target network at frame 1020000\n",
            "Updated target network at frame 1030000\n",
            "Updated target network at frame 1040000\n",
            "Updated target network at frame 1050000\n",
            "Updated target network at frame 1060000\n",
            "Updated target network at frame 1070000\n",
            "Updated target network at frame 1080000\n",
            "Episode 800\n",
            "  Average Reward (last 100): -16.52\n",
            "  Current Epsilon: 0.345\n",
            "  Average Loss: 0.0039\n",
            "  Frame Count: 1082335\n",
            "  Elapsed Time: 56.8 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -16.52\n",
            "Updated target network at frame 1090000\n",
            "Updated target network at frame 1100000\n",
            "Updated target network at frame 1110000\n",
            "Updated target network at frame 1120000\n",
            "Updated target network at frame 1130000\n",
            "Updated target network at frame 1140000\n",
            "Updated target network at frame 1150000\n",
            "Updated target network at frame 1160000\n",
            "Updated target network at frame 1170000\n",
            "Updated target network at frame 1180000\n",
            "Updated target network at frame 1190000\n",
            "Updated target network at frame 1200000\n",
            "Updated target network at frame 1210000\n",
            "Updated target network at frame 1220000\n",
            "Updated target network at frame 1230000\n",
            "Updated target network at frame 1240000\n",
            "Updated target network at frame 1250000\n",
            "Updated target network at frame 1260000\n",
            "Updated target network at frame 1270000\n",
            "Updated target network at frame 1280000\n",
            "Updated target network at frame 1290000\n",
            "Updated target network at frame 1300000\n",
            "Updated target network at frame 1310000\n",
            "Episode 900\n",
            "  Average Reward (last 100): -15.46\n",
            "  Current Epsilon: 0.276\n",
            "  Average Loss: 0.0040\n",
            "  Frame Count: 1312552\n",
            "  Elapsed Time: 68.6 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -15.46\n",
            "Updated target network at frame 1320000\n",
            "Updated target network at frame 1330000\n",
            "Updated target network at frame 1340000\n",
            "Updated target network at frame 1350000\n",
            "Updated target network at frame 1360000\n",
            "Updated target network at frame 1370000\n",
            "Updated target network at frame 1380000\n",
            "Updated target network at frame 1390000\n",
            "Updated target network at frame 1400000\n",
            "Updated target network at frame 1410000\n",
            "Updated target network at frame 1420000\n",
            "Updated target network at frame 1430000\n",
            "Updated target network at frame 1440000\n",
            "Updated target network at frame 1450000\n",
            "Updated target network at frame 1460000\n",
            "Updated target network at frame 1470000\n",
            "Updated target network at frame 1480000\n",
            "Updated target network at frame 1490000\n",
            "Updated target network at frame 1500000\n",
            "Updated target network at frame 1510000\n",
            "Updated target network at frame 1520000\n",
            "Updated target network at frame 1530000\n",
            "Updated target network at frame 1540000\n",
            "Updated target network at frame 1550000\n",
            "Updated target network at frame 1560000\n",
            "Updated target network at frame 1570000\n",
            "Episode 1000\n",
            "  Average Reward (last 100): -13.26\n",
            "  Current Epsilon: 0.215\n",
            "  Average Loss: 0.0037\n",
            "  Frame Count: 1575710\n",
            "  Elapsed Time: 82.5 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -13.26\n",
            "Updated target network at frame 1580000\n",
            "Updated target network at frame 1590000\n",
            "Updated target network at frame 1600000\n",
            "Updated target network at frame 1610000\n",
            "Updated target network at frame 1620000\n",
            "Updated target network at frame 1630000\n",
            "Updated target network at frame 1640000\n",
            "Updated target network at frame 1650000\n",
            "Updated target network at frame 1660000\n",
            "Updated target network at frame 1670000\n",
            "Updated target network at frame 1680000\n",
            "Updated target network at frame 1690000\n",
            "Updated target network at frame 1700000\n",
            "Updated target network at frame 1710000\n",
            "Updated target network at frame 1720000\n",
            "Updated target network at frame 1730000\n",
            "Updated target network at frame 1740000\n",
            "Updated target network at frame 1750000\n",
            "Updated target network at frame 1760000\n",
            "Updated target network at frame 1770000\n",
            "Updated target network at frame 1780000\n",
            "Updated target network at frame 1790000\n",
            "Updated target network at frame 1800000\n",
            "Updated target network at frame 1810000\n",
            "Updated target network at frame 1820000\n",
            "Updated target network at frame 1830000\n",
            "Updated target network at frame 1840000\n",
            "Updated target network at frame 1850000\n",
            "Updated target network at frame 1860000\n",
            "Updated target network at frame 1870000\n",
            "Episode 1100\n",
            "  Average Reward (last 100): -9.99\n",
            "  Current Epsilon: 0.162\n",
            "  Average Loss: 0.0047\n",
            "  Frame Count: 1873609\n",
            "  Elapsed Time: 98.3 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -9.99\n",
            "Updated target network at frame 1880000\n",
            "Updated target network at frame 1890000\n",
            "Updated target network at frame 1900000\n",
            "Updated target network at frame 1910000\n",
            "Updated target network at frame 1920000\n",
            "Updated target network at frame 1930000\n",
            "Updated target network at frame 1940000\n",
            "Updated target network at frame 1950000\n",
            "Updated target network at frame 1960000\n",
            "Updated target network at frame 1970000\n",
            "Updated target network at frame 1980000\n",
            "Updated target network at frame 1990000\n",
            "Updated target network at frame 2000000\n",
            "Updated target network at frame 2010000\n",
            "Updated target network at frame 2020000\n",
            "Updated target network at frame 2030000\n",
            "Updated target network at frame 2040000\n",
            "Updated target network at frame 2050000\n",
            "Updated target network at frame 2060000\n",
            "Updated target network at frame 2070000\n",
            "Updated target network at frame 2080000\n",
            "Updated target network at frame 2090000\n",
            "Updated target network at frame 2100000\n",
            "Updated target network at frame 2110000\n",
            "Updated target network at frame 2120000\n",
            "Updated target network at frame 2130000\n",
            "Updated target network at frame 2140000\n",
            "Updated target network at frame 2150000\n",
            "Updated target network at frame 2160000\n",
            "Updated target network at frame 2170000\n",
            "Updated target network at frame 2180000\n",
            "Updated target network at frame 2190000\n",
            "Updated target network at frame 2200000\n",
            "Episode 1200\n",
            "  Average Reward (last 100): -2.10\n",
            "  Current Epsilon: 0.119\n",
            "  Average Loss: 0.0041\n",
            "  Frame Count: 2202198\n",
            "  Elapsed Time: 115.7 min\n",
            "  Replay Buffer Size: 25000\n",
            "--------------------------------------------------\n",
            "New best model saved! Average reward: -2.10\n",
            "Updated target network at frame 2210000\n",
            "Updated target network at frame 2220000\n",
            "Updated target network at frame 2230000\n",
            "Updated target network at frame 2240000\n"
]

def final_training_report():
    extracted_data = []
    current_episode_data = {}

    for line in log_data:
        line = line.strip()

        if line.startswith("Episode"):
            if current_episode_data:
                extracted_data.append(current_episode_data)
                current_episode_data = {}
            episode_match = re.search(r"Episode (\d+)", line)
            if episode_match:
                current_episode_data["Episode"] = int(episode_match.group(1))
        elif "Average Reward" in line:
            reward_match = re.search(r"Average Reward \(last \d+\): (-?\d+\.?\d*)", line)
            if reward_match:
                current_episode_data["Average Reward"] = float(reward_match.group(1))
        elif "Current Epsilon" in line:
            epsilon_match = re.search(r"Current Epsilon: (\d+\.?\d*)", line)
            if epsilon_match:
                current_episode_data["Current Epsilon"] = float(epsilon_match.group(1))
        elif "Average Loss" in line:
            loss_match = re.search(r"Average Loss: (\d+\.?\d*)", line)
            if loss_match:
                current_episode_data["Average Loss"] = float(loss_match.group(1))
        elif "Frame Count" in line:
            frame_match = re.search(r"Frame Count: (\d+)", line)
            if frame_match:
                current_episode_data["Frame Count"] = int(frame_match.group(1))
        elif "Elapsed Time" in line:
            time_match = re.search(r"Elapsed Time: (\d+\.?\d*) min", line)
            if time_match:
                current_episode_data["Elapsed Time (min)"] = float(time_match.group(1))
        elif "Replay Buffer Size" in line:
            buffer_match = re.search(r"Replay Buffer Size: (\d+)", line)
            if buffer_match:
                current_episode_data["Replay Buffer Size"] = int(buffer_match.group(1))
        elif "--------------------------------------------------" in line:
            if current_episode_data and "Episode" in current_episode_data:
                if current_episode_data not in extracted_data:
                    extracted_data.append(current_episode_data)
                current_episode_data = {}
        if line == log_data[-1].strip() and current_episode_data and "Episode" in current_episode_data:
            if current_episode_data not in extracted_data:
                extracted_data.append(current_episode_data)

    df = pd.DataFrame(extracted_data)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df["Episode"], df["Average Reward"], marker='o', linestyle='-')
    plt.title("Average Reward Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(df["Episode"], df["Current Epsilon"], marker='o', linestyle='-', color='green')
    plt.title("Epsilon Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Current Epsilon")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(df["Episode"], df["Average Loss"], marker='o', linestyle='-', color='red')
    plt.title("Average Loss Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
