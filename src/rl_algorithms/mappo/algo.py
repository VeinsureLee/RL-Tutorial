"""
MAPPO：Multi-Agent PPO（Independent 版本）。

每个 agent 一套独立的 Actor-Critic 网络，独立收集 rollout，独立更新。
对外接口：``take_action(states) / update(agent_id, batch) / save / load``。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.mappo.net import ActorCriticNet


def _build_actor_critic(env, hidden_dim: int) -> ActorCriticNet:
    """按 env 尺寸构建 MAPPO 中 per-agent 的 Actor-Critic 网络。"""
    return ActorCriticNet(
        state_num=env.n_states,
        action_dim=env.n_actions,
        rows=env.rows,
        cols=env.cols,
        embedding_dim=64,
        hidden_dim=hidden_dim,
    )


class MAPPOBuffer:
    """MAPPO per-agent on-policy 轨迹缓冲区。"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.targets = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, target, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.targets.append(target)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.targets.clear()
        self.dones.clear()

    def get(self):
        """返回所有数据的 numpy array 元组。"""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.targets, dtype=np.int64),
            np.array(self.dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.states)


class MAPPO:
    """Multi-Agent PPO：每 agent 一套 Actor-Critic / buffer / optimizer。"""

    def __init__(self, env,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01, value_coef: float = 0.5,
                 num_epochs: int = 10, batch_size: int = 64,
                 hidden_dim: int = 128,
                 device: torch.device = torch.device("cpu")):
        self.env = env
        self.num_agents = env.num_agents
        self.n_actions = env.n_actions
        self.n_powers = env.n_powers
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        # 保留字段以兼容 trainer 接口
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0
        self.update_freq = 1000000

        self.nets = []
        self.optimizers = []
        self.buffers = []
        for _ in range(self.num_agents):
            net = _build_actor_critic(env, hidden_dim).to(device)
            self.nets.append(net)
            self.optimizers.append(optim.Adam(net.parameters(), lr=lr))
            self.buffers.append(MAPPOBuffer())

    def _decode_dir(self, action: int) -> int:
        return action // self.n_powers

    def take_action(self, states, training: bool = True):
        """
        独立采样：每个 agent 独立采样动作。
        训练时返回 (action_list, value_list, log_prob_list)，
        测试时只返回 action_list。
        """
        if training:
            actions = []
            values = []
            log_probs = []
            for i in range(self.num_agents):
                if self.env.done_flags is not None and self.env.done_flags[i]:
                    actions.append(0)
                    values.append(0.0)
                    log_probs.append(0.0)
                    continue
                target_idx = self.env.pos_to_index(*self.env.target_states[i])
                s = torch.tensor([states[i]], dtype=torch.long, device=self.device)
                t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    action, log_prob, value = self.nets[i].get_action(s, t)
                actions.append(int(action.item()))
                values.append(float(value.item()))
                log_probs.append(float(log_prob.item()))
            return actions, values, log_probs
        else:
            actions = []
            for i in range(self.num_agents):
                if self.env.done_flags is not None and self.env.done_flags[i]:
                    actions.append(0)
                    continue
                target_idx = self.env.pos_to_index(*self.env.target_states[i])
                s = torch.tensor([states[i]], dtype=torch.long, device=self.device)
                t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits, _ = self.nets[i](s, t)
                    action = torch.argmax(logits, dim=-1)
                actions.append(int(action.item()))
            return actions

    def _compute_returns_and_advantages(self, agent_id, rewards, values, dones, next_value):
        """计算 GAE advantage 和 discounted returns。"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, agent_id: int, next_value: float):
        """使用指定 agent 的 buffer 数据进行 PPO 更新。"""
        states, actions, rewards, values, old_log_probs, targets, dones = self.buffers[agent_id].get()
        if len(states) == 0:
            return 0.0

        returns, advantages = self._compute_returns_and_advantages(agent_id, rewards, values, dones, next_value)

        states_t = torch.tensor(states, dtype=torch.long, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=self.device)

        total_loss = 0.0
        num_updates = 0

        indices = np.arange(len(states))
        for _ in range(self.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                log_probs, values_pred, entropy = self.nets[agent_id].evaluate_actions(
                    states_t[batch_idx], targets_t[batch_idx], actions_t[batch_idx]
                )

                ratio = torch.exp(log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t[batch_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values_pred, returns_t[batch_idx])
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizers[agent_id].zero_grad()
                loss.backward()
                self.optimizers[agent_id].step()

                total_loss += float(loss.item())
                num_updates += 1

        self.buffers[agent_id].clear()
        return total_loss / max(num_updates, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {}
        for i in range(self.num_agents):
            ckpt[f"net_{i}"] = self.nets[i].state_dict()
            ckpt[f"optimizer_{i}"] = self.optimizers[i].state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.nets[i].load_state_dict(ckpt[f"net_{i}"])
            self.optimizers[i].load_state_dict(ckpt[f"optimizer_{i}"])
