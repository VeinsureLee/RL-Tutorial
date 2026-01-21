import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from rl_algorithms.agent import Agent
from env.env import Env


class Qnet(nn.Module):
    """
    Q网络，输入x和y，输出action value估计
    """
    def __init__(self, x_dim, y_dim, hidden_dim = 128, action_dim = 5):
        super(Qnet, self).__init__()
        self.fcx = nn.Linear(x_dim, hidden_dim)
        self.fcy = nn.Linear(y_dim, hidden_dim)
        # 拼接 x、y 后维度翻倍
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x, y):
        x = F.relu(self.fcx(x))
        y = F.relu(self.fcy(y))
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ReplayBuffer:
    """
    ReplayBuffer，用于存储经验
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    
class DQN(Agent):
    def __init__(self, env,
                 lr = 0.001, gamma = 0.99, 
                 epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.995,
                 num_episodes = 10, episode_length = 400000,
                 batch_size = 64, mini_batch_size = 32,
                 hidden_dim = 128, update_freq = 100):
        super().__init__(env=env, 
                         lr=lr, gamma=gamma, 
                         epsilon=epsilon, epsilon_min=epsilon_min, 
                         epsilon_decay=epsilon_decay, num_episodes=num_episodes, episode_length=episode_length)
        
        # 初始化经验回放缓冲区
        self.buffer = ReplayBuffer(episode_length * num_episodes * env.num_agents)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        
        self.hidden_dim = hidden_dim
        # 初始化Q网络和目标Q网络
        self.qnet = Qnet(x_dim = env.x_dim, y_dim = env.y_dim, hidden_dim = hidden_dim, action_dim = env.num_actions)
        self.target_qnet = Qnet(x_dim = env.x_dim, y_dim = env.y_dim, hidden_dim = hidden_dim, action_dim = env.num_actions)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()
        
        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.qnet.parameters(), lr = lr)
        self.loss_fn = nn.MSELoss()
        
        # 初始化设备
        self.device = torch.device("cpu")
        self.qnet.to(self.device)
        self.target_qnet.to(self.device)
        
        self.update_freq = update_freq
    
    def _state_to_tensor(self, states: np.ndarray):
        """
        将形如 (batch, 2) 的状态数组转换为 one-hot 张量
        """
        x_tensor = F.one_hot(torch.tensor(states[:, 0], dtype=torch.long, device=self.device), num_classes=self.env.x_dim).float()
        y_tensor = F.one_hot(torch.tensor(states[:, 1], dtype=torch.long, device=self.device), num_classes=self.env.y_dim).float()
        return x_tensor, y_tensor
    
    def take_action(self, state, agent_id = None, training = True):
        """
        选择动作
        :param state: 状态
        :param agent_id: agent_id
        :param training: 是否处于训练模式
        :return: action value
        """
        if agent_id is None:
            agent_id = 0
        target = self.env.target_states[agent_id]
        if state == target:
            # 到达目标后保持不动（找到对应索引，否则默认第0个动作）
            stay_idx = 0
            if (0, 0) in self.env.action_space:
                stay_idx = self.env.action_space.index((0, 0))
            return stay_idx
        else:
            # epsilon-greedy 策略
            if training and random.random() < self.epsilon:
                return random.randrange(self.env.num_actions)
            
            state_arr = np.array([[int(state[0]), int(state[1])]], dtype=np.int64)
            x_tensor, y_tensor = self._state_to_tensor(state_arr)
            with torch.no_grad():
                action_values = self.qnet(x_tensor, y_tensor)
            return action_values.argmax(dim=1).item()
    
    def update(self):
        """
        从ReplayBuffer采样批量经验，更新QNet及policy
        """
        batch = self.buffer.sample(self.mini_batch_size)
        states = np.array([b[0] for b in batch], dtype=np.int64)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.int64)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        x_tensor, y_tensor = self._state_to_tensor(states)
        x_next_tensor, y_next_tensor = self._state_to_tensor(next_states)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 当前Q值
        q_values = self.qnet(x_tensor, y_tensor)
        q_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_qnet(x_next_tensor, y_next_tensor)
            max_next_q_values, _ = next_q_values.max(dim=1)
            td_targets = reward_tensor + self.gamma * max_next_q_values * (1 - done_tensor)

        # 损失与优化
        loss = self.loss_fn(q_values, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_target_qnet(self):
        """同步目标Q网络参数"""
        self.target_qnet.load_state_dict(self.qnet.state_dict())
    
    def train(self):
        with tqdm(range(1, self.num_episodes + 1), desc="Training", unit='episode') as pbar:
            for global_episode in pbar:
                states, _ = self.env.reset()
                current_state = states[0] if isinstance(states, list) else states
                episode_reward = 0
                step_count = 0

                while step_count < self.episode_length:
                    action_idx = self.take_action(current_state, agent_id=0, training=True)
                    action = self.env.action_space[action_idx]
                    next_state, reward, done, _ = self.env.step(action)

                    episode_reward += reward

                    # 存入经验
                    self.buffer.add(current_state, action_idx, reward, next_state, done)

                    # 更新Q网络
                    if self.buffer.size() >= self.mini_batch_size:
                        self.update()

                    # 更新当前状态
                    current_state = next_state
                    # epsilon 衰减
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    step_count += 1

                    # 同步目标网络
                    if step_count % self.update_freq == 0:
                        self.update_target_qnet()

                    if done:
                        break

                if global_episode % 10 == 0:
                    pbar.set_postfix({
                        'epsilon': f"{self.epsilon:.3f}",
                        'return': f"{episode_reward:.3f}"
                    })
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "qnet": self.qnet.state_dict(),
            "target_qnet": self.target_qnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)
    
    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到模型文件: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(checkpoint["qnet"])
        self.target_qnet.load_state_dict(checkpoint.get("target_qnet", checkpoint["qnet"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)

if __name__ == "__main__":
    env = Env()
    dqn = DQN(env, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
              epsilon_decay=0.995, batch_size=64, mini_batch_size=400000, hidden_dim=128, 
              num_episodes=50, episode_length=1000, update_freq=50)
    dqn.train()
    dqn.save("models/dqn_model.pth")