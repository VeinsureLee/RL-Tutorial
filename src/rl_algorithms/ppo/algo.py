"""
PPO：单 agent Proximal Policy Optimization。

只训练 agent_id 的策略，其它 agent 在 trainer / tester 内由均匀随机动作填充。
对外接口：``take_action(state) / update(batch) / save / load``。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_algorithms.ppo.net import ActorCriticNet


def _build_actor_critic(env, hidden_dim: int, num_agents: int = 1) -> ActorCriticNet:
    """按 env 尺寸构建 PPO 的 Actor-Critic 网络。"""
    return ActorCriticNet(
        state_num=env.n_states,
        action_dim=env.n_actions,
        rows=env.rows,
        cols=env.cols,
        embedding_dim=64,
        hidden_dim=hidden_dim,
        num_agents=num_agents,
    )


class PPOBuffer:
    """PPO on-policy 轨迹缓冲区：存储完整 rollout 的 (s, a, r, v, logp, start, target, done)。"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.targets = []
        self.starts = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, start, target, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.starts.append(start)
        self.targets.append(target)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.starts.clear()
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
            np.array(self.starts, dtype=np.int64),
            np.array(self.targets, dtype=np.int64),
            np.array(self.dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.states)


class PPO:
    """单 agent PPO：只训练 agent_id 的策略，其它 agent 给随机动作。"""

    def __init__(self, env, agent_id: int = 0,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.05, value_coef: float = 0.5,
                 num_epochs: int = 10, batch_size: int = 64,
                 hidden_dim: int = 128,
                 max_grad_norm: float = 0.5, target_kl: float = 0.02,
                 entropy_coef_min: float = 0.001, entropy_coef_decay: float = 0.97,
                 test_temperature: float = 0.3,
                 device: torch.device = torch.device("cpu")):
        self.env = env
        self.agent_id = agent_id
        self.n_actions = env.n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # 稳定性参数：梯度裁剪上限 & KL 早停阈值
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        # 熵系数衰减：类比 DQN 的 epsilon 衰减，从探索过渡到确定
        # entropy_coef 起始值高 → 鼓励高熵分布 → 广泛探索
        # 每 ep 乘以 entropy_coef_decay，下界 entropy_coef_min → 后期分布塌缩近确定
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_decay = entropy_coef_decay
        # 测试时的 softmax 温度：<1 让分布更锐利接近 argmax，但仍保留少量随机性以破死循环
        self.test_temperature = test_temperature
        self.device = device

        # PPO 不需要 epsilon-greedy，保留字段以兼容 trainer 的日志打印
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0
        self.update_freq = 1000000  # 占位，不实际使用

        self.net = _build_actor_critic(env, hidden_dim, num_agents=1).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = PPOBuffer()

        # 记录当前 episode 的 start_idx，在 reset 时更新
        self.current_start_idx = None

    def take_action(self, state, training: bool = True):
        """采样动作，训练时返回 (action, value, log_prob)，测试时只返回 action。

        测试时按 softmax(logits / test_temperature) 采样：
        - test_temperature=1.0 等价训练时的随机策略，路径偏长（detour 多）
        - test_temperature<1 锐化分布，绝大部分时间走最优动作，少量随机性破死循环
        - test_temperature→0 等价 argmax，会在某些 state 反复选同一次优动作而卡死
        """
        target_idx = self.env.pos_to_index(*self.env.target_states[self.agent_id])
        if self.current_start_idx is None:
            self.current_start_idx = self.env.pos_to_index(*self.env.start_states[self.agent_id])

        s = torch.tensor([state], dtype=torch.long, device=self.device)
        t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
        start = torch.tensor([self.current_start_idx], dtype=torch.long, device=self.device)

        with torch.no_grad():
            if training:
                action, log_prob, value = self.net.get_action(s, t, start)
                return int(action.item()), float(value.item()), float(log_prob.item())
            else:
                logits, _ = self.net(s, t, start)
                probs = F.softmax(logits / self.test_temperature, dim=-1)
                action = torch.distributions.Categorical(probs).sample()
                return int(action.item())

    def reset_start_idx(self):
        """在环境 reset 时调用，更新当前 start_idx。"""
        self.current_start_idx = self.env.pos_to_index(*self.env.start_states[self.agent_id])

    def _compute_returns_and_advantages(self, rewards, values, dones, next_value):
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

    def update(self, next_value: float):
        """使用当前 buffer 中的 rollout 数据进行 PPO 更新。"""
        states, actions, rewards, values, old_log_probs, starts, targets, dones = self.buffer.get()
        if len(states) == 0:
            return 0.0

        returns, advantages = self._compute_returns_and_advantages(rewards, values, dones, next_value)

        states_t = torch.tensor(states, dtype=torch.long, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        starts_t = torch.tensor(starts, dtype=torch.long, device=self.device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=self.device)

        total_loss = 0.0
        num_updates = 0
        early_stop = False

        indices = np.arange(len(states))
        for _ in range(self.num_epochs):
            np.random.shuffle(indices)
            for batch_start in range(0, len(states), self.batch_size):
                end = batch_start + self.batch_size
                batch_idx = indices[batch_start:end]

                log_probs, values_pred, entropy = self.net.evaluate_actions(
                    states_t[batch_idx], targets_t[batch_idx], starts_t[batch_idx], actions_t[batch_idx]
                )

                # KL 早停：单 batch 近似 KL 超阈值即停止本轮全部更新，避免策略漂移过远
                with torch.no_grad():
                    approx_kl = (old_log_probs_t[batch_idx] - log_probs).mean().item()
                if approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

                ratio = torch.exp(log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value clip：限制 critic 单步预测变化幅度，防止 outlier returns 引发梯度爆炸
                v_pred_clipped = old_values_t[batch_idx] + (values_pred - old_values_t[batch_idx]).clamp(
                    -self.clip_epsilon, self.clip_epsilon
                )
                v_loss_unclipped = (values_pred - returns_t[batch_idx]) ** 2
                v_loss_clipped = (v_pred_clipped - returns_t[batch_idx]) ** 2
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = -entropy.mean()
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：防止极端 batch 的大梯度把策略一次性推坏
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                num_updates += 1
            if early_stop:
                break

        self.buffer.clear()
        return total_loss / max(num_updates, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
