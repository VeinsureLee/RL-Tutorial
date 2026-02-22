"""
MADQN 模型模块：仅包含 MADQN 智能体类。
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


class MADQN(Agent):
    def __init__(self,
                 env, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9,
                 num_episodes=5, episode_length=35000,
                 iteration=5, batch_size=64, mini_batch_size=64,
                 hidden_dim=128, update_freq=10, device=torch.device("cpu")):
        super().__init__(env=env,
                         lr=lr, gamma=gamma,
                         epsilon=epsilon, epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay,
                         num_episodes=num_episodes, episode_length=episode_length)

        self.num_agents = env.num_agents
        self.iteration = iteration

        self.buffer = [ReplayBuffer(episode_length * num_episodes * env.num_agents)
                       for _ in range(self.num_agents)]
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.hidden_dim = hidden_dim

        state_num = env.state_num
        self.q_nets = [Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                            action_dim=env.num_actions, x_dim=env.x_dim, y_dim=env.y_dim)
                       for _ in range(self.num_agents)]
        self.target_q_nets = [Qnet(state_num=state_num, embedding_dim=64, hidden_dim=hidden_dim,
                                   action_dim=env.num_actions, x_dim=env.x_dim, y_dim=env.y_dim)
                              for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
            self.target_q_nets[i].eval()

        self.optimizers = [optim.Adam(self.q_nets[i].parameters(), lr=lr)
                           for i in range(self.num_agents)]
        self.loss_fn = nn.MSELoss()
        self.device = device
        for i in range(self.num_agents):
            self.q_nets[i].to(self.device)
            self.target_q_nets[i].to(self.device)
        self.update_freq = update_freq

    def take_action(self, states, training=True):
        actions = []
        for agent_id in range(self.num_agents):
            target = self.env.target_states[agent_id]
            state = states[agent_id]
            state_idx_tensor = state_to_idx_tensor(state, self.env.y_dim, self.device)
            target_idx_tensor = state_to_idx_tensor(target, self.env.y_dim, self.device)

            with torch.no_grad():
                action_values = self.q_nets[agent_id](state_idx_tensor, target_idx_tensor)
            optimal_action = action_values.argmax(dim=1).item()

            if training:
                num_actions = self.env.num_actions
                probs = torch.full((num_actions,), self.epsilon / num_actions, device=self.device)
                probs[optimal_action] = 1 - self.epsilon + self.epsilon / num_actions
                dist = torch.distributions.Categorical(probs=probs)
                actions.append(dist.sample().item())
            else:
                actions.append(optimal_action)
        return actions

    def update(self, agent_id):
        if len(self.buffer[agent_id]) < self.mini_batch_size:
            return None

        batch = self.buffer[agent_id].sample(self.mini_batch_size)

        def normalize_state(s):
            if isinstance(s, np.ndarray):
                s = s.flatten()
                if len(s) >= 2:
                    return [int(s[0]), int(s[1])]
            elif isinstance(s, (tuple, list)):
                if len(s) >= 2:
                    first_elem = s[0]
                    if isinstance(first_elem, (tuple, list, np.ndarray)):
                        return normalize_state(first_elem)
                    return [int(s[0]), int(s[1])]
            try:
                s_array = np.array(s).flatten()
                if len(s_array) >= 2:
                    return [int(s_array[0]), int(s_array[1])]
            except Exception:
                pass
            return [0, 0]

        states = np.array([normalize_state(b[0]) for b in batch], dtype=np.int64)
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([normalize_state(b[3]) for b in batch], dtype=np.int64)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        state_idx = state_to_idx_tensor(states, self.env.y_dim, self.device)
        next_state_idx = state_to_idx_tensor(next_states, self.env.y_dim, self.device)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(rewards, device=self.device)
        target = self.env.target_states[agent_id]
        target_idx = state_to_idx_tensor(target, self.env.y_dim, self.device).repeat(len(batch))

        q = self.q_nets[agent_id](state_idx, target_idx)
        q = q.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_q_nets[agent_id](next_state_idx, target_idx)
            max_next_q = next_q.max(dim=1)[0]
            td_target = reward_tensor + self.gamma * max_next_q * (1 - torch.tensor(dones, device=self.device))
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
        print(f"模型已保存到: {path}")

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到模型文件: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.q_nets[i].load_state_dict(checkpoint[f"qnet_{i}"])
            self.target_q_nets[i].load_state_dict(checkpoint.get(f"target_qnet_{i}", checkpoint[f"qnet_{i}"]))
            self.optimizers[i].load_state_dict(checkpoint[f"optimizer_{i}"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
