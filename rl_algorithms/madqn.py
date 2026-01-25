import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.env import Env
from rl_algorithms.agent import Agent
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from typing import List, Tuple, Any, Optional
from collections import deque
import random
import numpy as np



class Qnet(nn.Module):
    """
    Q(s, a | target)

    target 通过 embedding 输入网络
    相对位置信息作为主干
    """

    def __init__(
        self,
        state_num: int,
        action_dim: int,
        x_dim: int,
        y_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        # ===== state embedding =====
        self.embedding = nn.Embedding(state_num, embedding_dim)

        self.abs_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
        )

        # ===== target embedding branch =====
        self.target_fc = nn.Sequential(
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
        # 融合 state embedding, target embedding 和 relative position (64 + 64 + 64 = 192)
        self.head = nn.Sequential(
            nn.Linear(192, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_idx: torch.Tensor, target_idx: torch.Tensor):
        """
        state_idx: (B,)
        target_idx: (B,)
        """

        # ===== absolute state embedding =====
        state_emb = self.embedding(state_idx)
        h_abs = self.abs_fc(state_emb)

        # ===== target embedding =====
        target_emb = self.embedding(target_idx)
        h_target = self.target_fc(target_emb)

        # ===== relative position to target =====
        sx = state_idx // self.y_dim
        sy = state_idx % self.y_dim

        tx = target_idx // self.y_dim
        ty = target_idx % self.y_dim

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
        h = torch.cat([h_abs, h_target, h_rel], dim=1)
        q = self.head(h)

        return q


class ReplayBuffer:
    """
    ReplayBuffer，用于存储经验
    """

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, agent_id, state, action, reward, next_state, done, target):
        self.buffer.append(
            (agent_id, state, action, reward, next_state, done, target))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class MADQN(Agent):
    def __init__(self, env, agent_id=None,
                 lr=0.001, gamma=0.99, iteration=10,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 num_episodes=10, episode_length=400000,
                 batch_size=64, mini_batch_size=32,
                 hidden_dim=128, update_freq=100):
        super().__init__(env=env,
                         lr=lr, gamma=gamma,
                         epsilon=epsilon, epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay, num_episodes=num_episodes, episode_length=episode_length)

        self.agent_id = agent_id
        self.iteration = iteration

        # 初始化经验回放缓冲区
        self.buffer = ReplayBuffer(
            episode_length * num_episodes * env.num_agents)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        self.hidden_dim = hidden_dim

        # 初始化Q网络和目标Q网络
        state_num = env.state_num  # 总状态数
        self.qnet = Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                         action_dim=env.num_actions, x_dim=env.x_dim, y_dim=env.y_dim)
        self.target_qnet = Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                                action_dim=env.num_actions, x_dim=env.x_dim, y_dim=env.y_dim)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
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
            state_idx_tensor = torch.tensor(
                [state_idx], dtype=torch.long, device=self.device)
        else:
            # 批量状态 (batch, 2)
            state = np.array(state, dtype=np.int64)
            state_indices = state[:, 0] * self.env.y_dim + state[:, 1]
            state_idx_tensor = torch.tensor(
                state_indices, dtype=torch.long, device=self.device)
        return state_idx_tensor

    def take_action(self, state, agent_id=None, training=True):
        """
        选择动作
        :param state: 状态
        :param agent_id: 机器人编号，如果为None则使用self.agent_id
        :param training: 是否处于训练模式
        :return: action索引（int）
        """
        if agent_id is None:
            if self.agent_id is None:
                agent_id = 0
            else:
                agent_id = self.agent_id
        target = self.env.target_states[agent_id]
        if state == target:
            # 到达目标后保持不动，返回停留动作的索引
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
                probs = torch.full(
                    (num_actions,), self.epsilon / num_actions, device=self.device)
                probs[optimal_action] = 1 - self.epsilon + \
                    self.epsilon / num_actions

                # 根据概率分布采样动作
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample().item()
                return action
            else:
                # 测试模式：直接返回最优动作
                return optimal_action

    def update(self):
        batch = self.buffer.sample(self.mini_batch_size)

        agent_ids = np.array([b[0] for b in batch])
        states = np.array([b[1] for b in batch])
        actions = np.array([b[2] for b in batch])
        rewards = np.array([b[3] for b in batch], dtype=np.float32)
        next_states = np.array([b[4] for b in batch])
        dones = np.array([b[5] for b in batch], dtype=np.float32)
        targets = np.array([b[6] for b in batch])

        state_idx = self._state_to_idx_tensor(states)
        next_state_idx = self._state_to_idx_tensor(next_states)
        target_idx = self._state_to_idx_tensor(targets)

        action_tensor = torch.tensor(
            actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, device=self.device)

        q = self.qnet(state_idx, target_idx)
        q = q.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_qnet(next_state_idx, target_idx)
            max_next_q = next_q.max(dim=1)[0]
            td_target = reward_tensor + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_qnet(self):
        """同步目标Q网络参数"""
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def train(self):
        # Train all agents collectively
        print(f"Begin to train MADQN (all agents), iteration: {self.iteration}")
        epsilon = self.epsilon
        for i in range(self.iteration):
            self.epsilon = epsilon
            # 使用tqdm创建进度条
            pbar = tqdm(range(1, self.num_episodes + 1),
                        desc=f"Iteration({i+1}) progress".format(i+1), unit="episode")

            for ep in pbar:
                states, _ = self.env.reset()
                targets = self.env.target_states

                ep_return = 0  # 所有agent的总回报

                for t in range(self.episode_length):
                    # 为所有agent选择动作（使用MADQN策略）
                    actions = []
                    action_indices = []
                    for agent_id in range(self.env.num_agents):
                        state_i = states[agent_id]
                        action_idx = self.take_action(state_i, agent_id=agent_id, training=True)
                        action = self.env.action_space[action_idx]
                        actions.append(action)
                        action_indices.append(action_idx)

                    # 执行所有agent的动作
                    next_states, rewards, dones, _ = self.env.step(actions)

                    # 为所有agent存储经验并更新
                    for agent_id in range(self.env.num_agents):
                        state_i = states[agent_id]
                        action_idx = action_indices[agent_id]
                        reward = rewards[agent_id]
                        next_state = next_states[agent_id]
                        done = dones[agent_id]
                        target = targets[agent_id]

                        ep_return += reward

                        self.buffer.add(
                            agent_id,
                            state_i,
                            action_idx,
                            reward,
                            next_state,
                            done,
                            target
                        )

                    # 如果buffer中有足够的经验，进行更新
                    if len(self.buffer) >= self.mini_batch_size:
                        self.update()

                    # 更新状态
                    states = next_states

                    # 定期更新目标网络
                    if t % self.update_freq == 0:
                        self.update_target_qnet()

                    # 如果所有agent都完成，提前结束
                    if all(dones):
                        break

                # epsilon **按 episode 衰减**
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon * self.epsilon_decay
                )

                # 更新进度条显示信息
                pbar.set_postfix({
                    'Episode': ep,
                    'Return': f'{ep_return:.2f}',
                    'Epsilon': f'{self.epsilon:.3f}'
                })

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "qnet": self.qnet.state_dict(),
            "target_qnet": self.target_qnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)
        print(f"模型已保存到: {path}")

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到模型文件: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(checkpoint["qnet"])
        self.target_qnet.load_state_dict(
            checkpoint.get("target_qnet", checkpoint["qnet"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)


if __name__ == "__main__":
    env = Env()
    madqn = MADQN(
        env,
        agent_id=0,
        lr=1e-3,
        gamma=0.99,
        iteration=5,
        epsilon=0.8,
        epsilon_decay=0.95,   # 按 episode
        epsilon_min=0.1,
        num_episodes=50,
        episode_length=35000,
        mini_batch_size=64,
        update_freq=10
    )

    madqn.train()
    madqn.save("models/madqn_model.pth")
