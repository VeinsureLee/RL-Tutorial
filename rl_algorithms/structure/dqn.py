"""
DQN 模型模块：仅包含 DQN 智能体类。
训练与绘图见 train、plot。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.utils.agent import Agent
from rl_algorithms.net.qnet import Qnet
from rl_algorithms.utils.replaybuffer import ReplayBuffer
from rl_algorithms.utils.utils import state_to_idx_tensor


class DQN(Agent):
    def __init__(self, env, agent_id=None,
                 lr=0.001, gamma=0.99, iteration=10,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 num_episodes=10, episode_length=400000,
                 batch_size=64, mini_batch_size=32,
                 hidden_dim=128, update_freq=100, device=torch.device("cpu")):
        super().__init__(env=env,
                         lr=lr, gamma=gamma,
                         epsilon=epsilon, epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay, num_episodes=num_episodes, episode_length=episode_length)

        self.agent_id = agent_id
        self.iteration = iteration

        self.buffer = ReplayBuffer(episode_length * num_episodes * env.num_agents)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.hidden_dim = hidden_dim

        state_num = env.state_num
        self.qnet = Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                         action_dim=env.num_actions, x_dim=env.x_dim, y_dim=env.y_dim)
        self.target_qnet = Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                                action_dim=env.num_actions, x_dim=env.x_dim, y_dim=env.y_dim)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.device = device
        self.qnet.to(self.device)
        self.target_qnet.to(self.device)
        self.update_freq = update_freq

    def take_action(self, state, training=True):
        if self.agent_id is None:
            self.agent_id = 0
        target = self.env.target_states[self.agent_id]
        if state == target:
            return 0
        state_idx_tensor = state_to_idx_tensor(state, self.env.y_dim, self.device)
        target_idx_tensor = state_to_idx_tensor(target, self.env.y_dim, self.device)

        with torch.no_grad():
            action_values = self.qnet(state_idx_tensor, target_idx_tensor)
        optimal_action = action_values.argmax(dim=1).item()

        if training:
            num_actions = self.env.num_actions
            probs = torch.full((num_actions,), self.epsilon / num_actions, device=self.device)
            probs[optimal_action] = 1 - self.epsilon + self.epsilon / num_actions
            dist = torch.distributions.Categorical(probs=probs)
            return dist.sample().item()
        return optimal_action

    def update(self):
        batch = self.buffer.sample(self.mini_batch_size)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        state_idx = state_to_idx_tensor(states, self.env.y_dim, self.device)
        next_state_idx = state_to_idx_tensor(next_states, self.env.y_dim, self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        target = self.env.target_states[self.agent_id]
        target_idx_single = state_to_idx_tensor(target, self.env.y_dim, self.device)
        target_idx = target_idx_single.repeat(len(batch))

        q = self.qnet(state_idx, target_idx)
        q = q.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_qnet(next_state_idx, target_idx)
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
        print(f"模型已保存到: {path}")

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到模型文件: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(checkpoint["qnet"])
        self.target_qnet.load_state_dict(checkpoint.get("target_qnet", checkpoint["qnet"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
