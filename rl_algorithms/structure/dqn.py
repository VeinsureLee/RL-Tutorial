"""
DQN 模型模块：单 agent DQN，对齐论文。
动作空间 = n_dirs x n_powers 的复合动作。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.net.qnet import Qnet
from rl_algorithms.utils.replaybuffer import ReplayBuffer


class DQN:
    def __init__(self, env, agent_id=0,
                 lr=1e-4, gamma=0.9,
                 epsilon=0.1, epsilon_min=0.1, epsilon_decay=1.0,
                 hidden_dim=128, update_freq=100,
                 replay_buffer_size=50000, device=torch.device("cpu")):
        self.env = env
        self.agent_id = agent_id
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

        self.qnet = Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                         action_dim=self.n_actions, x_dim=env.rows, y_dim=env.cols).to(device)
        self.target_qnet = Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                                action_dim=self.n_actions, x_dim=env.rows, y_dim=env.cols).to(device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = 128

    def decode_action(self, action):
        return action // self.n_powers, action % self.n_powers

    def take_action(self, state, training=True):
        """
        Args:
            state: state_index (int)

        Returns:
            action: compound action index (int)
        """
        target_idx = self.env.pos_to_index(*self.env.target_states[self.agent_id])
        state_tensor = torch.tensor([state], dtype=torch.long, device=self.device)
        target_tensor = torch.tensor([target_idx], dtype=torch.long, device=self.device)

        with torch.no_grad():
            q_values = self.qnet(state_tensor, target_tensor)

        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return q_values.argmax(dim=1).item()

    def update(self, batch):
        """
        Args:
            batch: (states, actions, rewards, next_states, dones) numpy arrays
        """
        states, actions_arr, rewards, next_states, dones = batch

        state_idx = torch.tensor(states, dtype=torch.long, device=self.device)
        next_state_idx = torch.tensor(next_states, dtype=torch.long, device=self.device)
        action_tensor = torch.tensor(actions_arr, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        target_state = self.env.target_states[self.agent_id]
        target_idx = self.env.pos_to_index(*target_state)
        target_tensor = torch.full((len(states),), target_idx, dtype=torch.long, device=self.device)

        q = self.qnet(state_idx, target_tensor)
        q = q.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_qnet(next_state_idx, target_tensor)
            max_next_q = next_q.max(dim=1)[0]
            td_target = reward_tensor + self.gamma * max_next_q * (1 - dones_tensor)

        loss = self.loss_fn(q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

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
            raise FileNotFoundError(f"未找到模型文件: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.qnet.load_state_dict(checkpoint["qnet"])
        self.target_qnet.load_state_dict(checkpoint.get("target_qnet", checkpoint["qnet"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
