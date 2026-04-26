"""
DQN：单 agent 学习。

只训练 ``agent_id`` 的策略，其它 agent 在 trainer / tester 内由均匀随机动作填充。
对外接口：``take_action(state) / update(batch) / save / load``。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.dqn.qnet import Qnet
from rl_algorithms.replay import ReplayBuffer


def _build_qnet(env, hidden_dim: int) -> Qnet:
    """按 env 尺寸构建 DQN 的 Q 网络。"""
    return Qnet(
        state_num=env.n_states,
        action_dim=env.n_actions,
        rows=env.rows,
        cols=env.cols,
        embedding_dim=64,
        hidden_dim=hidden_dim,
    )


class DQN:
    """单 agent DQN：只训练 agent_id 的策略，其它 agent 由训练循环给随机动作。"""

    def __init__(self, env, agent_id: int = 0,
                 lr: float = 1e-4, gamma: float = 0.9,
                 epsilon: float = 0.5, epsilon_min: float = 0.01, epsilon_decay: float = 0.99,
                 hidden_dim: int = 128, update_freq: int = 100,
                 replay_buffer_size: int = 50000,
                 device: torch.device = torch.device("cpu")):
        self.env = env
        self.agent_id = agent_id
        self.n_actions = env.n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.update_freq = update_freq

        self.qnet = _build_qnet(env, hidden_dim).to(device)
        self.target_qnet = _build_qnet(env, hidden_dim).to(device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = 128

    def take_action(self, state, training: bool = True) -> int:
        """epsilon-greedy。state 为单个整数 state_index。"""
        target_idx = self.env.pos_to_index(*self.env.target_states[self.agent_id])
        s = torch.tensor([state], dtype=torch.long, device=self.device)
        t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.qnet(s, t)
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(q.argmax(dim=1).item())

    def update(self, batch) -> float:
        """单步 TD(0) 更新。batch=(states, actions, rewards, next_states, dones)。"""
        states, actions, rewards, next_states, dones = batch
        s = torch.tensor(states, dtype=torch.long, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.long, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)
        target_idx = self.env.pos_to_index(*self.env.target_states[self.agent_id])
        t = torch.full((len(states),), target_idx, dtype=torch.long, device=self.device)

        q = self.qnet(s, t).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_qnet(ns, t).max(dim=1)[0]
            td_target = r + self.gamma * max_next_q * (1 - d)
        loss = self.loss_fn(q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target_qnet(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

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
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.qnet.load_state_dict(ckpt["qnet"])
        self.target_qnet.load_state_dict(ckpt.get("target_qnet", ckpt["qnet"]))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
