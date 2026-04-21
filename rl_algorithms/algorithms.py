"""
DQN 与 MADQN 算法：统一 take_action / update / save / load 接口。

DQN  : 单 agent 学习，指定 agent_id；其他 agent 训练/测试时走随机策略（在 trainer 内拼接）。
MADQN: Independent DQN，每个 agent 独立 Q 网络、目标网络、回放缓冲。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.qnet import Qnet
from rl_algorithms.replay import ReplayBuffer


def _build_qnet(env, hidden_dim: int) -> Qnet:
    """按 env 尺寸构建 Q 网络。"""
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


class MADQN:
    """Independent DQN：每 agent 一套 Q/target/buffer/optimizer，共享 env。"""

    def __init__(self, env,
                 lr: float = 1e-4, gamma: float = 0.9,
                 epsilon: float = 0.5, epsilon_min: float = 0.01, epsilon_decay: float = 0.99,
                 hidden_dim: int = 128, update_freq: int = 100,
                 replay_buffer_size: int = 50000,
                 device: torch.device = torch.device("cpu")):
        self.env = env
        self.num_agents = env.num_agents
        self.n_actions = env.n_actions
        self.n_powers = env.n_powers
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.update_freq = update_freq

        self.q_nets = []
        self.target_q_nets = []
        self.optimizers = []
        self.buffers = []
        for _ in range(self.num_agents):
            q = _build_qnet(env, hidden_dim).to(device)
            t = _build_qnet(env, hidden_dim).to(device)
            t.load_state_dict(q.state_dict())
            t.eval()
            self.q_nets.append(q)
            self.target_q_nets.append(t)
            self.optimizers.append(optim.Adam(q.parameters(), lr=lr))
            self.buffers.append(ReplayBuffer(replay_buffer_size))
        self.loss_fn = nn.MSELoss()
        self.batch_size = 128

    def _decode_dir(self, action: int) -> int:
        return action // self.n_powers

    def take_action(self, states, training: bool = True):
        """
        对每 agent epsilon-greedy 选动作；对方向分量做冲突规避：
        若下一格已被其它 agent 选或是禁区，则按 Q 排序尝试替代动作。
        """
        actions = []
        occupied = set()
        directions = self.env.directions
        for i in range(self.num_agents):
            if self.env.done_flags is not None and self.env.done_flags[i]:
                actions.append(0)
                continue
            target_idx = self.env.pos_to_index(*self.env.target_states[i])
            s = torch.tensor([states[i]], dtype=torch.long, device=self.device)
            t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
            with torch.no_grad():
                q = self.q_nets[i](s, t)
            if training and np.random.random() < self.epsilon:
                action = int(np.random.randint(self.n_actions))
            else:
                action = int(q.argmax(dim=1).item())

            dr, dc = directions[self._decode_dir(action)]
            cur_r, cur_c = self.env.positions[i]
            new_r = int(cur_r + dr)
            new_c = int(cur_c + dc)
            if (new_r, new_c) in occupied or (new_r, new_c) in self.env.forbidden_set:
                sorted_actions = q.argsort(dim=1, descending=True).squeeze().tolist()
                if isinstance(sorted_actions, int):
                    sorted_actions = [sorted_actions]
                for alt in sorted_actions:
                    dr2, dc2 = directions[self._decode_dir(alt)]
                    nr2 = int(cur_r + dr2)
                    nc2 = int(cur_c + dc2)
                    if (nr2, nc2) not in occupied and (nr2, nc2) not in self.env.forbidden_set:
                        action, new_r, new_c = alt, nr2, nc2
                        break
            occupied.add((new_r, new_c))
            actions.append(action)
        return actions

    def update(self, agent_id: int, batch) -> float:
        states, actions, rewards, next_states, dones = batch
        s = torch.tensor(states, dtype=torch.long, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.long, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)
        target_idx = self.env.pos_to_index(*self.env.target_states[agent_id])
        t = torch.full((len(states),), target_idx, dtype=torch.long, device=self.device)

        q = self.q_nets[agent_id](s, t).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_q_nets[agent_id](ns, t).max(dim=1)[0]
            td_target = r + self.gamma * max_next_q * (1 - d)
        loss = self.loss_fn(q, td_target)
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()
        return float(loss.item())

    def update_target_qnet(self, agent_id: int):
        self.target_q_nets[agent_id].load_state_dict(self.q_nets[agent_id].state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {"epsilon": self.epsilon}
        for i in range(self.num_agents):
            ckpt[f"qnet_{i}"] = self.q_nets[i].state_dict()
            ckpt[f"target_qnet_{i}"] = self.target_q_nets[i].state_dict()
            ckpt[f"optimizer_{i}"] = self.optimizers[i].state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.q_nets[i].load_state_dict(ckpt[f"qnet_{i}"])
            self.target_q_nets[i].load_state_dict(ckpt.get(f"target_qnet_{i}", ckpt[f"qnet_{i}"]))
            self.optimizers[i].load_state_dict(ckpt[f"optimizer_{i}"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
