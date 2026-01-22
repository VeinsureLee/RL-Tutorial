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
    Q网络，输入state idx和target state idx，使用embedding，输出action value估计
    加入相对位置信息以引导向目标靠近
    """
    def __init__(self, state_num, embedding_dim = 64, hidden_dim = 128, action_dim = 5, x_dim = None, y_dim = None):
        super(Qnet, self).__init__()
        self.embedding = nn.Embedding(state_num, embedding_dim)
        self.target_embedding = nn.Embedding(state_num, embedding_dim)
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        # 相对位置特征维度：dx, dy, distance (3维)
        relative_dim = 3
        # 拼接 state embedding, target embedding 和相对位置特征
        self.fc1 = nn.Linear(embedding_dim * 2 + relative_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def _idx_to_coord(self, state_idx, x_dim, y_dim):
        """将 state_idx 转换为坐标 (x, y)"""
        x = state_idx // y_dim
        y = state_idx % y_dim
        return x, y
        
    def forward(self, state_idx, target_state_idx):
        """
        :param state_idx: state索引，形状为 (batch_size,) 的 long tensor
        :param target_state_idx: target state索引，形状为 (batch_size,) 的 long tensor
        :return: action value，形状为 (batch_size, action_dim)
        """
        state_emb = self.embedding(state_idx)
        target_emb = self.target_embedding(target_state_idx)
        
        # 计算相对位置信息（引导向目标靠近）
        if self.x_dim is not None and self.y_dim is not None:
            # 将 idx 转换为坐标
            state_x = state_idx // self.y_dim
            state_y = state_idx % self.y_dim
            target_x = target_state_idx // self.y_dim
            target_y = target_state_idx % self.y_dim
            
            # 计算相对位置：dx, dy, distance
            dx = (target_x - state_x).float() / self.x_dim  # 归一化
            dy = (target_y - state_y).float() / self.y_dim  # 归一化
            distance = torch.sqrt(dx**2 + dy**2 + 1e-8)  # 避免除零
            
            relative_features = torch.stack([dx, dy, distance], dim=1)
        else:
            # 如果没有提供 x_dim 和 y_dim，则不使用相对位置信息
            relative_features = torch.zeros(state_idx.shape[0], 3, device=state_idx.device)
        
        # 拼接 state embedding, target embedding 和相对位置特征
        x = torch.cat((state_emb, target_emb, relative_features), dim=1)
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
        state_num = env.state_num # 总状态数
        self.qnet = Qnet(state_num = state_num, embedding_dim = 64, hidden_dim = hidden_dim, 
                         action_dim = env.num_actions, x_dim = env.x_dim, y_dim = env.y_dim)
        self.target_qnet = Qnet(state_num = state_num, embedding_dim = 64, hidden_dim = hidden_dim, 
                                action_dim = env.num_actions, x_dim = env.x_dim, y_dim = env.y_dim)
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
    
    def _state_to_idx_tensor(self, state):
        """
        将状态转换为 state idx 张量
        :param state: 单个状态 (x, y) 或批量状态数组 (batch, 2)
        :return: state idx tensor，单个状态返回形状 (1,)，批量状态返回形状 (batch,)
        """
        if isinstance(state, (tuple, list, np.ndarray)) and len(state) == 2 and isinstance(state[0], (int, np.integer)):
            # 单个状态 (x, y)
            state_idx = int(state[0]) * self.env.y_dim + int(state[1])
            state_idx_tensor = torch.tensor([state_idx], dtype=torch.long, device=self.device)
        else:
            # 批量状态 (batch, 2)
            state = np.array(state, dtype=np.int64)
            state_indices = state[:, 0] * self.env.y_dim + state[:, 1]
            state_idx_tensor = torch.tensor(state_indices, dtype=torch.long, device=self.device)
        return state_idx_tensor
    
    def take_action(self, state, agent_id = None, training = True):
        """
        选择动作
        :param state: 状态
        :param agent_id: 机器人编号
        :param training: 是否处于训练模式
        :return: action索引（int）
        """
        if agent_id is None:
            agent_id = 0
        target = self.env.target_states[agent_id]
        if state == target:
            # 到达目标后保持不动，返回停留动作的索引
            stay_action = (0, 0)
            if stay_action in self.env.action_space:
                return self.env.action_space.index(stay_action)
            else:
                return 0  # 默认返回第0个动作
        else:
            # 将 (x, y) 转换为 state_idx: state_idx = x * y_dim + y
            state_idx_tensor = self._state_to_idx_tensor(state)
            target_idx_tensor = self._state_to_idx_tensor(target)
            
            with torch.no_grad():
                action_values = self.qnet(state_idx_tensor, target_idx_tensor)
            
            # 获取最优动作
            optimal_action = action_values.argmax(dim=1).item()
            
            if training:
                # 概率分配：最优动作概率最大，其他动作概率相同
                # 最优动作概率：1 - epsilon + epsilon/num_actions
                # 其他动作概率：epsilon/num_actions
                num_actions = self.env.num_actions
                probs = torch.full((num_actions,), self.epsilon / num_actions, device=self.device)
                probs[optimal_action] = 1 - self.epsilon + self.epsilon / num_actions
                
                # 根据概率分布采样动作
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample().item()
                return action
            else:
                # 测试模式：直接返回最优动作
                return optimal_action
    
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

        # 使用 _state_to_idx_tensor 将状态转换为 state_idx
        state_idx_tensor = self._state_to_idx_tensor(states)
        next_state_idx_tensor = self._state_to_idx_tensor(next_states)
        # 获取 target state（假设所有样本都是 agent_id=0）
        target_state = self.env.target_states[0]
        target_state_idx = int(target_state[0]) * self.env.y_dim + int(target_state[1])
        # 创建与 batch_size 相同大小的 target_state_idx tensor
        target_state_idx_tensor = torch.full((len(states),), target_state_idx, dtype=torch.long, device=self.device)
        
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 当前Q值
        q_values = self.qnet(state_idx_tensor, target_state_idx_tensor)
        q_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_qnet(next_state_idx_tensor, target_state_idx_tensor)
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
                    # 如果已经到达目标，直接结束episode
                    if current_state == self.env.target_states[0]:
                        # 到达目标，给予目标奖励
                        episode_reward += self.env.reward_target
                        break
                    
                    # take_action 现在总是返回动作索引
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
                        'return': f"{episode_reward:.3f}",
                        'steps': f"{step_count}",
                        'buffer': f"{self.buffer.size()}"
                    })
                
                # 添加调试信息：如果前几个episode的reward都是0，打印详细信息
                if global_episode <= 3:
                    print(f"\nEpisode {global_episode}: reward={episode_reward:.3f}, steps={step_count}, "
                          f"start={states[0] if isinstance(states, list) else states}, "
                          f"target={self.env.target_states[0]}, buffer_size={self.buffer.size()}")
    
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
    env.reward_target = 100
    env.reward_forbidden = -5
    env.reward_step = -1
    dqn = DQN(env, lr=0.001, gamma=1, epsilon=1.0, epsilon_min=0.01, 
              epsilon_decay=0.995, batch_size=64, mini_batch_size=200, hidden_dim=128, 
              num_episodes=50, episode_length=1000, update_freq=50)
    dqn.train()
    dqn.save("models/dqn_model.pth")