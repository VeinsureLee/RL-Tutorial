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


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn


class Qnet(nn.Module):
    """
    Q(s, a | fixed target)

    target 是常数，不作为网络输入
    相对位置信息作为主干
    """

    def __init__(
        self,
        state_num: int,
        action_dim: int,
        x_dim: int,
        y_dim: int,
        target_xy: tuple,          # (tx, ty)
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        # ===== 固定 target（不参与训练）=====
        target = torch.tensor(target_xy, dtype=torch.long)
        self.register_buffer("target_xy", target)

        # ===== state embedding（辅助）=====
        self.embedding = nn.Embedding(state_num, embedding_dim)

        self.abs_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
        )

        # ===== relative position branch（主干）=====
        self.rel_fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # ===== fusion head =====
        self.head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_idx: torch.Tensor):
        """
        state_idx: (B,)
        """

        # ===== absolute state embedding =====
        state_emb = self.embedding(state_idx)
        h_abs = self.abs_fc(state_emb)

        # ===== relative position to fixed target =====
        sx = state_idx // self.y_dim
        sy = state_idx % self.y_dim

        tx, ty = self.target_xy
        tx = tx.expand_as(sx)
        ty = ty.expand_as(sy)

        dx = (tx - sx).float()
        dy = (ty - sy).float()

        dx_n = dx / self.x_dim
        dy_n = dy / self.y_dim

        abs_dx = torch.abs(dx_n)
        abs_dy = torch.abs(dy_n)

        l1 = abs_dx + abs_dy
        l2 = torch.sqrt(dx_n ** 2 + dy_n ** 2 + 1e-8)

        sign_dx = torch.sign(dx_n)
        sign_dy = torch.sign(dy_n)

        rel_feat = torch.stack(
            [dx_n, dy_n, abs_dx, abs_dy, l1, l2, sign_dx, sign_dy],
            dim=1,
        )

        h_rel = self.rel_fc(rel_feat)

        # ===== fusion =====
        h = torch.cat([h_abs, h_rel], dim=1)
        q = self.head(h)

        return q



    
class ReplayBuffer:
    """
    ReplayBuffer，用于存储经验
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        
    def add(self, state, action, reward, next_state, done, target):
        self.buffer.append((state, action, reward, next_state, done, target))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
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
        # 获取第一个agent的target作为固定target（Qnet设计为固定target）
        target_xy = env.target_states[0]  # (x, y) 元组
        self.qnet = Qnet(state_num = state_num, embedding_dim = 64, hidden_dim = hidden_dim, 
                         action_dim = env.num_actions, x_dim = env.x_dim, y_dim = env.y_dim,
                         target_xy = target_xy)
        self.target_qnet = Qnet(state_num = state_num, embedding_dim = 64, hidden_dim = hidden_dim, 
                                action_dim = env.num_actions, x_dim = env.x_dim, y_dim = env.y_dim,
                                target_xy = target_xy)
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
            
            with torch.no_grad():
                action_values = self.qnet(state_idx_tensor)
            
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
        batch = self.buffer.sample(self.mini_batch_size)

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        targets = np.array([b[5] for b in batch])

        state_idx = self._state_to_idx_tensor(states)
        next_state_idx = self._state_to_idx_tensor(next_states)

        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, device=self.device)

        q = self.qnet(state_idx)
        q = q.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_qnet(next_state_idx)
            max_next_q = next_q.max(dim=1)[0]
            td_target = reward_tensor + self.gamma * max_next_q

        loss = self.loss_fn(q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    
    def update_target_qnet(self):
        """同步目标Q网络参数"""
        self.target_qnet.load_state_dict(self.qnet.state_dict())
    
    def train(self):
        for ep in range(1, self.num_episodes + 1):
            states, _ = self.env.reset()
            state = states[0]
            target = self.env.target_states[0]

            ep_return = 0

            for t in range(self.episode_length):
                action_idx = self.take_action(state, agent_id=0, training=True)
                action = self.env.action_space[action_idx]

                next_state, reward, done, _ = self.env.step(action)

                ep_return += reward

                self.buffer.add(
                    state,
                    action_idx,
                    reward,
                    next_state,
                    done,
                    target
                )

                if len(self.buffer) >= self.mini_batch_size:
                    self.update()

                state = next_state

                if t % self.update_freq == 0:
                    self.update_target_qnet()

                if done:
                    break

            # epsilon **按 episode 衰减**
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            print(f"Episode {ep} | return={ep_return:.2f} | epsilon={self.epsilon:.3f}")

        
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
    env.reward_forbidden = -1
    env.reward_step = -1
    dqn =DQN(
                env,
                lr=1e-3,
                gamma=0.99,
                epsilon=1,
                epsilon_decay=0.95,   # 按 episode
                epsilon_min=0.4,
                num_episodes=50,
                episode_length=4000,
                mini_batch_size=64,
                update_freq=200
            )

    dqn.train()
    dqn.save("models/dqn_model.pth")