"""
MADQN 模型模块：Independent DQN for multi-agent，对齐论文。
动作空间 = n_dirs x n_powers 的复合动作。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.net.qnet import Qnet
from rl_algorithms.utils.replaybuffer import ReplayBuffer


class MADQN:
    def __init__(self, env, lr=1e-4, gamma=0.9,
                 epsilon=0.1, epsilon_min=0.1, epsilon_decay=1.0,
                 hidden_dim=128, update_freq=100,
                 replay_buffer_size=50000, device=torch.device("cpu")):
        self.env = env
        self.num_agents = env.num_agents
        self.n_actions = env.n_actions
        self.n_dirs = env.n_dirs
        self.n_powers = env.n_powers
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.update_freq = update_freq

        state_num = env.n_states
        self.map_cols = env.cols

        self.q_nets = []
        self.target_q_nets = []
        self.optimizers = []
        self.buffers = []

        for _ in range(self.num_agents):
            q = Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                     action_dim=self.n_actions, x_dim=env.rows, y_dim=env.cols).to(device)
            t = Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                     action_dim=self.n_actions, x_dim=env.rows, y_dim=env.cols).to(device)
            t.load_state_dict(q.state_dict())
            t.eval()
            self.q_nets.append(q)
            self.target_q_nets.append(t)
            self.optimizers.append(optim.Adam(q.parameters(), lr=lr))
            self.buffers.append(ReplayBuffer(replay_buffer_size))

        self.loss_fn = nn.MSELoss()
        self.batch_size = 128  # will be overridden by training loop

    def decode_action(self, action):
        """compound action -> (dir_idx, power_idx)"""
        return action // self.n_powers, action % self.n_powers

    def take_action(self, states, training=True):
        """
        Epsilon-greedy with collision avoidance on direction component.

        Args:
            states: list of state_index (int)

        Returns:
            actions: list of compound action indices
        """
        actions = []
        occupied = set()  # 已选定的下一步位置
        directions = self.env.directions

        for i in range(self.num_agents):
            # 已到达目标的 agent
            if self.env.done_flags is not None and self.env.done_flags[i]:
                actions.append(0)
                continue

            target_idx = self.env.pos_to_index(*self.env.target_states[i])
            state_tensor = torch.tensor([states[i]], dtype=torch.long, device=self.device)
            target_tensor = torch.tensor([target_idx], dtype=torch.long, device=self.device)

            with torch.no_grad():
                q_values = self.q_nets[i](state_tensor, target_tensor)

            if training and np.random.random() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = q_values.argmax(dim=1).item()

            # 检查方向是否导致冲突
            dir_idx, power_idx = self.decode_action(action)
            dr, dc = directions[dir_idx]
            cur_r, cur_c = self.env.positions[i]
            new_r = int(cur_r + dr)
            new_c = int(cur_c + dc)

            if (new_r, new_c) in occupied or (new_r, new_c) in self.env.forbidden_set:
                # 按 Q 值排序尝试其他动作
                sorted_actions = q_values.argsort(dim=1, descending=True).squeeze().tolist()
                if isinstance(sorted_actions, int):
                    sorted_actions = [sorted_actions]
                for alt_action in sorted_actions:
                    d_idx, _ = self.decode_action(alt_action)
                    dr2, dc2 = directions[d_idx]
                    nr2 = int(cur_r + dr2)
                    nc2 = int(cur_c + dc2)
                    if (nr2, nc2) not in occupied and (nr2, nc2) not in self.env.forbidden_set:
                        action = alt_action
                        new_r, new_c = nr2, nc2
                        break

            occupied.add((new_r, new_c))
            actions.append(action)

        return actions

    def update(self, agent_id, batch):
        """
        更新指定 agent 的 Q 网络。

        Args:
            agent_id: agent 索引
            batch: (states, actions, rewards, next_states, dones) numpy arrays
        """
        states, actions_arr, rewards, next_states, dones = batch

        state_idx = torch.tensor(states, dtype=torch.long, device=self.device)
        next_state_idx = torch.tensor(next_states, dtype=torch.long, device=self.device)
        action_tensor = torch.tensor(actions_arr, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        target_state = self.env.target_states[agent_id]
        target_idx = self.env.pos_to_index(*target_state)
        target_tensor = torch.full((len(states),), target_idx, dtype=torch.long, device=self.device)

        q = self.q_nets[agent_id](state_idx, target_tensor)
        q = q.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_q_nets[agent_id](next_state_idx, target_tensor)
            max_next_q = next_q.max(dim=1)[0]
            td_target = reward_tensor + self.gamma * max_next_q * (1 - dones_tensor)

        loss = self.loss_fn(q, td_target)
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()
        return loss.item()

    def update_target_qnet(self, agent_id):
        self.target_q_nets[agent_id].load_state_dict(self.q_nets[agent_id].state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {"epsilon": self.epsilon}
        for i in range(self.num_agents):
            checkpoint[f"qnet_{i}"] = self.q_nets[i].state_dict()
            checkpoint[f"target_qnet_{i}"] = self.target_q_nets[i].state_dict()
            checkpoint[f"optimizer_{i}"] = self.optimizers[i].state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到模型文件: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.q_nets[i].load_state_dict(checkpoint[f"qnet_{i}"])
            self.target_q_nets[i].load_state_dict(checkpoint.get(f"target_qnet_{i}", checkpoint[f"qnet_{i}"]))
            self.optimizers[i].load_state_dict(checkpoint[f"optimizer_{i}"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
