"""
MAPPO：Multi-Agent PPO（CTDE 版本）。

每个 agent 一套独立的 Actor（分散执行），但每个 Critic 用全局信息（集中训练）。
对外接口：``take_action(states) / update(agent_id, batch) / save / load``。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_algorithms.mappo.net import ActorCriticNet


def _build_actor_critic(env, hidden_dim: int, num_agents: int) -> ActorCriticNet:
    """按 env 尺寸构建 MAPPO 中 per-agent 的 Actor-Critic 网络（CTDE）。"""
    return ActorCriticNet(
        state_num=env.n_states,
        action_dim=env.n_actions,
        rows=env.rows,
        cols=env.cols,
        num_agents=num_agents,
        embedding_dim=64,
        hidden_dim=hidden_dim,
    )


class MAPPOBuffer:
    """MAPPO per-agent on-policy 轨迹缓冲区（存储全局信息供 Critic 使用）。"""

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        # 局部信息（当前 agent）
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.targets = []
        self.starts = []
        self.agent_ids = []
        self.dones = []
        # 全局信息（所有 agent）
        self.all_states = []
        self.all_starts = []
        self.all_targets = []

    def add(self, state, action, reward, value, log_prob, start, target, agent_id, done,
            all_states, all_starts, all_targets):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.starts.append(start)
        self.targets.append(target)
        self.agent_ids.append(agent_id)
        self.dones.append(done)
        self.all_states.append(all_states.copy())
        self.all_starts.append(all_starts.copy())
        self.all_targets.append(all_targets.copy())

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.starts.clear()
        self.targets.clear()
        self.agent_ids.clear()
        self.dones.clear()
        self.all_states.clear()
        self.all_starts.clear()
        self.all_targets.clear()

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
            np.array(self.agent_ids, dtype=np.int64),
            np.array(self.dones, dtype=np.float32),
            np.array(self.all_states, dtype=np.int64),
            np.array(self.all_starts, dtype=np.int64),
            np.array(self.all_targets, dtype=np.int64),
        )

    def __len__(self):
        return len(self.states)


class MAPPO:
    """Multi-Agent PPO（CTDE）：per-agent Actor + per-agent centralized Critic。"""

    def __init__(self, env,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.05, value_coef: float = 0.5,
                 num_epochs: int = 10, batch_size: int = 64,
                 hidden_dim: int = 128,
                 max_grad_norm: float = 0.5, target_kl: float = 0.02,
                 entropy_coef_min: float = 0.001, entropy_coef_decay: float = 0.97,
                 test_temperature: float = 0.3,
                 reward_scale: float = 0.01,
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
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_decay = entropy_coef_decay
        self.test_temperature = test_temperature
        # 推入 buffer 的 reward 会乘上 reward_scale；critic value 也学的是 scaled return，
        # 因此 buffer 中存储的 (reward, value) 在同一 scale 上，GAE/value-loss 自洽。
        # 详见 trainer._push_on_policy_data MAPPO 分支。
        self.reward_scale = reward_scale
        self.device = device

        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0
        self.update_freq = 1000000

        # 参数共享：所有 agent 共用同一套 Actor-Critic 与 Adam，agent_idx 作为输入
        # 区分不同 agent。逻辑等价于"标准 MAPPO 中 actor/critic 双共享"配方。
        # buffer 仍然每 agent 独立（on-policy 数据按 agent 收集，update 时按 agent
        # 调用，每次都对同一份共享参数做一次梯度步）。
        shared_net = _build_actor_critic(env, hidden_dim, self.num_agents).to(device)
        shared_opt = optim.Adam(shared_net.parameters(), lr=lr)
        self.nets = [shared_net] * self.num_agents
        self.optimizers = [shared_opt] * self.num_agents
        self.buffers = [MAPPOBuffer(self.num_agents) for _ in range(self.num_agents)]
        # save/load 用的单独引用
        self._shared_net = shared_net
        self._shared_optimizer = shared_opt

        # 记录当前 episode 的 start_indices
        self.current_start_indices = None
        # 缓存全局信息供计算 value 使用
        self._cached_all_starts = None
        self._cached_all_targets = None

    def _decode_dir(self, action: int) -> int:
        return action // self.n_powers

    def compute_global_values(self, states):
        """用全局信息计算所有 agent 的 value（供 trainer 存入 buffer 使用）。"""
        if self.current_start_indices is None:
            self.reset_start_indices()

        all_states = np.array(states, dtype=np.int64)
        all_starts = np.array(self.current_start_indices, dtype=np.int64)
        all_targets = np.array([
            self.env.pos_to_index(*self.env.target_states[i])
            for i in range(self.num_agents)
        ], dtype=np.int64)

        values = []
        for i in range(self.num_agents):
            if self.env.done_flags is not None and self.env.done_flags[i]:
                values.append(0.0)
            else:
                all_states_t = torch.tensor(all_states, dtype=torch.long, device=self.device).unsqueeze(0)
                all_starts_t = torch.tensor(all_starts, dtype=torch.long, device=self.device).unsqueeze(0)
                all_targets_t = torch.tensor(all_targets, dtype=torch.long, device=self.device).unsqueeze(0)
                all_agent_idx_t = torch.tensor(list(range(self.num_agents)), dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    value = self.nets[i].forward_critic(
                        all_states_t, all_starts_t, all_targets_t, all_agent_idx_t
                    )
                values.append(float(value.item()))
        return values, all_states, all_starts, all_targets

    def take_action(self, states, training: bool = True):
        """
        独立采样：每个 agent 独立采样动作（Actor 只用局部信息）。
        训练时返回 (action_list, value_list, log_prob_list)，
        测试时只返回 action_list。
        """
        if self.current_start_indices is None:
            self.reset_start_indices()

        if training:
            actions = []
            values = []
            log_probs = []
            # 先计算全局 values（用 Critic 全局输入）
            global_values, all_states, all_starts, all_targets = self.compute_global_values(states)
            # 缓存全局信息供 push_to_buffer 使用
            self._cached_all_starts = all_starts
            self._cached_all_targets = all_targets

            for i in range(self.num_agents):
                if self.env.done_flags is not None and self.env.done_flags[i]:
                    actions.append(0)
                    values.append(0.0)
                    log_probs.append(0.0)
                    continue
                target_idx = self.env.pos_to_index(*self.env.target_states[i])
                start_idx = self.current_start_indices[i]
                s = torch.tensor([states[i]], dtype=torch.long, device=self.device)
                t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
                start = torch.tensor([start_idx], dtype=torch.long, device=self.device)
                agent_idx = torch.tensor([i], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.nets[i].forward_actor(s, t, start, agent_idx)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                actions.append(int(action.item()))
                values.append(global_values[i])
                log_probs.append(float(log_prob.item()))
            return actions, values, log_probs
        else:
            actions = []
            for i in range(self.num_agents):
                if self.env.done_flags is not None and self.env.done_flags[i]:
                    actions.append(0)
                    continue
                target_idx = self.env.pos_to_index(*self.env.target_states[i])
                start_idx = self.current_start_indices[i]
                s = torch.tensor([states[i]], dtype=torch.long, device=self.device)
                t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
                start = torch.tensor([start_idx], dtype=torch.long, device=self.device)
                agent_idx = torch.tensor([i], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.nets[i].forward_actor(s, t, start, agent_idx)
                    probs = F.softmax(logits / self.test_temperature, dim=-1)
                    action = torch.distributions.Categorical(probs).sample()
                actions.append(int(action.item()))
            return actions

    def _gae(self, rewards, values, dones, next_value):
        """只算 GAE advantages 和 returns，不做归一化（归一化留到全局拼接后做）。"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        T = len(rewards)
        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_advantage = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            )
        returns = advantages + values
        return returns, advantages

    def update_all(self, next_values) -> float:
        """合并所有 agent 的 rollout，做一次 PPO update。

        - 各 agent 用自己的 next_value 算 GAE（advantage / return）
        - 拼接所有 agent 的 transitions 为一个大 batch
        - 在拼接后做一次全局 advantage 归一化（避免单 agent 内 std 受卡死 ep 拉爆）
        - 用共享 net + 共享 optimizer 跑标准 PPO 多 epoch / minibatch 循环
        - 早停 / clip / value clip / 熵奖励均与单 agent 版一致
        - 最后清空所有 agent 的 buffer

        参数
        ----
        next_values : list[float]
            每个 agent 的 bootstrap value（done 的 agent 传 0）。
        """
        keys = ["states", "actions", "rewards", "values", "old_log_probs",
                "starts", "targets", "agent_ids", "dones",
                "all_states", "all_starts", "all_targets",
                "returns", "advantages"]
        bins = {k: [] for k in keys}

        for i in range(self.num_agents):
            (states, actions, rewards, values, old_log_probs,
             starts, targets, agent_ids, dones,
             all_states, all_starts, all_targets) = self.buffers[i].get()
            if len(states) == 0:
                continue
            returns, advantages = self._gae(rewards, values, dones, next_values[i])
            for k, arr in zip(keys, [states, actions, rewards, values, old_log_probs,
                                     starts, targets, agent_ids, dones,
                                     all_states, all_starts, all_targets,
                                     returns, advantages]):
                bins[k].append(arr)

        if not bins["states"]:
            for i in range(self.num_agents):
                self.buffers[i].clear()
            return 0.0

        cat = {k: np.concatenate(bins[k], axis=0) for k in keys}
        # 全局 advantage 归一化（跨 agent 拼接后做，比单 agent 内归一化稳）
        adv = cat["advantages"]
        cat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        device = self.device
        states_t = torch.tensor(cat["states"], dtype=torch.long, device=device)
        actions_t = torch.tensor(cat["actions"], dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(cat["old_log_probs"], dtype=torch.float32, device=device)
        old_values_t = torch.tensor(cat["values"], dtype=torch.float32, device=device)
        returns_t = torch.tensor(cat["returns"], dtype=torch.float32, device=device)
        advantages_t = torch.tensor(cat["advantages"], dtype=torch.float32, device=device)
        starts_t = torch.tensor(cat["starts"], dtype=torch.long, device=device)
        targets_t = torch.tensor(cat["targets"], dtype=torch.long, device=device)
        agent_ids_t = torch.tensor(cat["agent_ids"], dtype=torch.long, device=device)
        all_states_t = torch.tensor(cat["all_states"], dtype=torch.long, device=device)
        all_starts_t = torch.tensor(cat["all_starts"], dtype=torch.long, device=device)
        all_targets_t = torch.tensor(cat["all_targets"], dtype=torch.long, device=device)
        N = states_t.shape[0]
        all_agent_idx_t = torch.tensor([list(range(self.num_agents))] * N,
                                       dtype=torch.long, device=device)

        total_loss = 0.0
        num_updates = 0
        early_stop = False
        indices = np.arange(N)

        for _ in range(self.num_epochs):
            if early_stop:
                break
            np.random.shuffle(indices)
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                log_probs, values_pred, entropy = self._shared_net.evaluate_actions(
                    states_t[batch_idx], targets_t[batch_idx], starts_t[batch_idx],
                    actions_t[batch_idx], agent_ids_t[batch_idx],
                    all_states_t[batch_idx], all_starts_t[batch_idx],
                    all_targets_t[batch_idx], all_agent_idx_t[batch_idx]
                )

                with torch.no_grad():
                    approx_kl = (old_log_probs_t[batch_idx] - log_probs).mean().item()
                if approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

                ratio = torch.exp(log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                v_pred_clipped = old_values_t[batch_idx] + (values_pred - old_values_t[batch_idx]).clamp(
                    -self.clip_epsilon, self.clip_epsilon
                )
                v_loss_unclipped = (values_pred - returns_t[batch_idx]) ** 2
                v_loss_clipped = (v_pred_clipped - returns_t[batch_idx]) ** 2
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = -entropy.mean()
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self._shared_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._shared_net.parameters(), self.max_grad_norm)
                self._shared_optimizer.step()

                total_loss += float(loss.item())
                num_updates += 1

        for i in range(self.num_agents):
            self.buffers[i].clear()
        return total_loss / max(num_updates, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "net": self._shared_net.state_dict(),
            "optimizer": self._shared_optimizer.state_dict(),
            "num_agents": self.num_agents,
        }, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # 兼容旧的 per-agent ckpt：若有 net_0 字段则按旧格式只读 agent 0 的权重
        if "net" in ckpt:
            self._shared_net.load_state_dict(ckpt["net"])
            self._shared_optimizer.load_state_dict(ckpt["optimizer"])
        else:
            self._shared_net.load_state_dict(ckpt["net_0"])
            self._shared_optimizer.load_state_dict(ckpt["optimizer_0"])

    def reset_start_indices(self):
        """在环境 reset 时调用，更新当前 start_indices。"""
        self.current_start_indices = [
            self.env.pos_to_index(*self.env.start_states[i])
            for i in range(self.num_agents)
        ]
