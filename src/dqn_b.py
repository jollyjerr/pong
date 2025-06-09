import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import cv2

from collections import deque

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_w = self._conv_output_size(input_shape[1], 8, 4)
        conv_w = self._conv_output_size(conv_w, 4, 2)
        conv_w = self._conv_output_size(conv_w, 3, 1)
        conv_h = self._conv_output_size(input_shape[2], 8, 4)
        conv_h = self._conv_output_size(conv_h, 4, 2)
        conv_h = self._conv_output_size(conv_h, 3, 1)
        linear_input_size = conv_w * conv_h * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self._initialize_weights()
    
    def _conv_output_size(self, size, kernel_size, stride):
        return (size - kernel_size) // stride + 1
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def stack_frames(stacked_frames, frame, is_new_episode=False):
    frame = preprocess_frame(frame)
    
    if is_new_episode:
        stacked_frames = deque([frame for _ in range(4)], maxlen=4)
    else:
        stacked_frames.append(frame)
    
    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames
