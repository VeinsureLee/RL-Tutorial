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


class DQNNetwork(nn.Module):
    """
    DQN神经网络：输入状态，输出每个动作的Q值
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent类，继承Agent基类
    每个agent使用独立的DQN网络进行学习
    """
    
    def __init__(self, env, agent_id=0, lr=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000, 
                 batch_size=64, hidden_dim=128, device=None):
        """
        初始化DQN Agent
        
        :param env: 环境实例
        :param agent_id: agent的ID（用于多agent场景）
        :param lr: 学习率
        :param gamma: 折扣因子
        :param epsilon: 初始探索率
        :param epsilon_min: 最小探索率
        :param epsilon_decay: 探索率衰减
        :param memory_size: 经验回放缓冲区大小
        :param batch_size: 批次大小
        :param hidden_dim: 隐藏层维度
        :param device: 计算设备（'cuda'或'cpu'）
        """
        self.env = env
        self.agent_id = agent_id
        self.action_space = env.action_space
        self.num_actions = len(env.action_space)
        
        # 状态维度：当前坐标(x, y) + 目标坐标(target_x, target_y) = 4维
        self.state_dim = 4
        self.action_dim = self.num_actions
        
        # 超参数
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # 设备
        if device is None:
            # 强制使用CPU，避免CUDA兼容性问题
            # 如果未来需要GPU，可以检查CUDA capability是否支持
            self.device = torch.device("cpu")
            # 可选：如果确实需要GPU且版本支持，可以使用：
            # if torch.cuda.is_available():
            #     # 检查CUDA capability
            #     try:
            #         capability = torch.cuda.get_device_capability(0)
            #         if capability[0] >= 8:  # 支持sm_80及以上
            #             self.device = torch.device("cuda")
            #         else:
            #             self.device = torch.device("cpu")
            #     except:
            #         self.device = torch.device("cpu")
            # else:
            #     self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # 神经网络
        self.q_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 初始化目标网络
        self.update_target_network()
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(memory_size)
        
        # 训练计数器
        self.train_step = 0
        self.target_update_freq = 100  # 每100步更新一次目标网络
        
    def _state_to_input(self, state, target_state):
        """
        将状态转换为神经网络输入
        :param state: 当前状态 (x, y)
        :param target_state: 目标状态 (target_x, target_y)
        :return: 输入向量 [x, y, target_x, target_y]
        """
        x, y = state
        target_x, target_y = target_state
        # 归一化到[0, 1]范围
        max_x = self.env.grid_rows - 1
        max_y = self.env.grid_cols - 1
        return np.array([
            x / max_x if max_x > 0 else 0,
            y / max_y if max_y > 0 else 0,
            target_x / max_x if max_x > 0 else 0,
            target_y / max_y if max_y > 0 else 0
        ], dtype=np.float32)
    
    def _action_to_index(self, action):
        """将动作转换为索引"""
        return self.action_space.index(action)
    
    def _index_to_action(self, index):
        """将索引转换为动作"""
        return self.action_space[index]
    
    def select_action(self, state: Any, agent_id: Optional[int] = None, training: bool = True) -> Any:
        """
        使用epsilon-greedy策略选择动作
        
        :param state: 当前状态（可以是单个状态或状态列表）
        :param agent_id: agent的ID（可选）
        :param training: 是否处于训练模式
        :return: 选择的动作
        """
        # 获取目标状态
        if hasattr(self.env, 'target_states'):
            if isinstance(self.env.target_states, list):
                target_state = self.env.target_states[self.agent_id]
            else:
                target_state = self.env.target_states
        else:
            target_state = self.env.target_state
        
        # 如果是多agent环境且state是列表
        if isinstance(state, (list, tuple)) and len(state) == self.env.num_agents:
            # 只处理当前agent的状态
            state = state[self.agent_id]
        
        # Epsilon-greedy策略
        if training and random.random() < self.epsilon:
            # 随机探索
            return random.choice(self.action_space)
        else:
            # 利用：选择Q值最大的动作
            state_input = self._state_to_input(state, target_state)
            state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_index = q_values.argmax().item()
            
            return self._index_to_action(action_index)
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        # 获取目标状态
        if hasattr(self.env, 'target_states'):
            if isinstance(self.env.target_states, list):
                target_state = self.env.target_states[self.agent_id]
            else:
                target_state = self.env.target_states
        else:
            target_state = self.env.target_state
        
        # 转换状态为输入格式
        state_input = self._state_to_input(state, target_state)
        next_state_input = self._state_to_input(next_state, target_state)
        action_index = self._action_to_index(action)
        
        self.memory.push(state_input, action_index, reward, next_state_input, done)
    
    def train(self, batch_size=None):
        """
        训练DQN网络
        :param batch_size: 批次大小（如果为None，使用默认值）
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch_size = batch_size or self.batch_size
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 下一个状态的最大Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.train_step = checkpoint.get('train_step', 0)


class DQN:
    """
    DQN主类，管理多个DQN Agent（多agent场景）
    如果只有一个agent，也可以直接使用DQNAgent
    """
    
    def __init__(self, env, **kwargs):
        """
        初始化DQN
        :param env: 环境实例
        :param kwargs: 传递给DQNAgent的参数
        """
        self.env = env
        self.num_agents = env.num_agents
        
        # 为每个agent创建独立的DQN网络
        self.agents = []
        for agent_id in range(self.num_agents):
            agent = DQNAgent(env, agent_id=agent_id, **kwargs)
            self.agents.append(agent)
    
    def select_action(self, states, training=True):
        """
        为所有agent选择动作
        :param states: 状态列表（每个agent一个状态）
        :param training: 是否处于训练模式
        :return: 动作列表
        """
        actions = []
        for agent_id, agent in enumerate(self.agents):
            action = agent.select_action(states, agent_id=agent_id, training=training)
            actions.append(action)
        return actions
    
    def remember(self, states, actions, rewards, next_states, dones):
        """
        存储所有agent的经验
        """
        for agent_id, agent in enumerate(self.agents):
            agent.remember(states[agent_id], actions[agent_id], 
                          rewards[agent_id], next_states[agent_id], dones[agent_id])
    
    def train(self, batch_size=None):
        """
        训练所有agent
        :return: 平均损失
        """
        losses = []
        for agent in self.agents:
            loss = agent.train(batch_size)
            if loss is not None:
                losses.append(loss)
        return np.mean(losses) if losses else None
    
    def save(self, path_prefix: str):
        """保存所有agent的模型"""
        for agent_id, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent_{agent_id}.pth")
    
    def load(self, path_prefix: str):
        """加载所有agent的模型"""
        for agent_id, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent_{agent_id}.pth")
