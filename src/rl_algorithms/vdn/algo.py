"""
VDN（Value Decomposition Networks）：每 agent 独立 Q_i 网络 + 加法分解，CTDE 联合训练。

核心思想：Q_tot = sum_i Q_i(s_i, a_i)
加法分解满足 IGM 条件，执行时各 agent 独立 argmax Q_i。

与 QMIX 的区别：无 mixer、无全局状态；Q_tot 直接求和，结构更简洁。
与旧版（共享 Qnet）的区别：每 agent 拥有独立参数，消除梯度互扰——共享参数时两
    agent 的梯度在同一组权重上叠加，大奖励 agent 会虚抬小奖励 agent 的 Q 值，导
    致策略紊乱。独立参数后各 Q_i 仅由本 agent 的 (s_i, a_i) 对驱动，收敛更稳定。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.vdn.qnet import Qnet
from rl_algorithms.replay import JointReplayBuffer


def _build_qnet(env, hidden_dim: int) -> Qnet:
    return Qnet(
        state_num=env.n_states,
        action_dim=env.n_actions,
        rows=env.rows,
        cols=env.cols,
        embedding_dim=64,
        hidden_dim=hidden_dim,
    )


class VDN:
    """每 agent 独立 Q_i + 加法 Q_tot 分解，联合 TD 反传。"""

    def __init__(self, env,
                 lr: float = 1e-4, gamma: float = 0.9,
                 epsilon: float = 0.5, epsilon_min: float = 0.01, epsilon_decay: float = 0.99,
                 hidden_dim: int = 128, update_freq: int = 100,
                 replay_buffer_size: int = 50000,
                 device: torch.device = torch.device("cpu"),
                 **kwargs):
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

        # 每 agent 独立的 Q_i + target（放 ModuleList 方便统一取 parameters）
        self.q_nets = nn.ModuleList(
            [_build_qnet(env, hidden_dim) for _ in range(self.num_agents)]
        ).to(device)
        self.target_q_nets = nn.ModuleList(
            [_build_qnet(env, hidden_dim) for _ in range(self.num_agents)]
        ).to(device)
        for i in range(self.num_agents):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
            self.target_q_nets[i].eval()

        # 单一 optimizer 覆盖所有 Q_i 参数（无 mixer）
        self.optimizer = optim.Adam(self.q_nets.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = JointReplayBuffer(replay_buffer_size)
        self.batch_size = 128

    # ------------------------------------------------------------------ 工具
    def _decode_dir(self, action: int) -> int:
        return action // self.n_powers

    # ------------------------------------------------------------------ 动作选择
    def take_action(self, states, training: bool = True):
        """
        Decentralized epsilon-greedy + 冲突规避（越界 / 禁区 / 同格占用）。
        每 agent 使用自己独立的 Q_i 网络决策。
        """
        directions = self.env.directions
        rows, cols = self.env.rows, self.env.cols

        actions = []
        occupied = set()

        for i in range(self.num_agents):
            if self.env.done_flags is not None and self.env.done_flags[i]:
                actions.append(0)
                continue

            target_idx = self.env.pos_to_index(*self.env.target_states[i])
            s = torch.tensor([states[i]], dtype=torch.long, device=self.device)
            t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
            with torch.no_grad():
                q = self.q_nets[i](s, t)  # (1, A)

            if training and np.random.random() < self.epsilon:
                action = int(np.random.randint(self.n_actions))
            else:
                action = int(q.argmax(dim=1).item())

            dr, dc = directions[self._decode_dir(action)]
            cur_r, cur_c = self.env.positions[i]
            new_r = int(cur_r + dr)
            new_c = int(cur_c + dc)

            # 越界 / 禁区 / 占用 统一判定
            def _invalid(r, c):
                return (not (0 <= r < rows and 0 <= c < cols)
                        or (r, c) in occupied
                        or (r, c) in self.env.forbidden_set)

            if _invalid(new_r, new_c):
                sorted_actions = q.argsort(dim=1, descending=True).squeeze().tolist()
                if isinstance(sorted_actions, int):
                    sorted_actions = [sorted_actions]
                for alt in sorted_actions:
                    dr2, dc2 = directions[self._decode_dir(alt)]
                    nr2 = int(cur_r + dr2)
                    nc2 = int(cur_c + dc2)
                    if not _invalid(nr2, nc2):
                        action, new_r, new_c = alt, nr2, nc2
                        break

            occupied.add((new_r, new_c))
            actions.append(action)
        return actions

    # ------------------------------------------------------------------ 联合更新
    def update(self, batch) -> float:
        """
        每 agent 独立 Q_i forward，加法汇总得 Q_tot，对联合 MSE 反传。
        梯度仅经由本 agent 的 Q_i 参数传播，各 agent 间无梯度互扰。

        Post-terminal 修正：当 s_i == target_i（agent 已停在终点）时，把该 agent
        的 Q_i 在当前 Q_tot 中清零。
        原因：done agent 停在终点后 action 固定为 0，Q_i(target, 0) 若为非零值会
        通过联合损失持续偏移其他 agent 的梯度，使其他 agent 的 Q 值被虚高或虚低，
        导致原地转圈。终点步本身（s_i = 终点前一格 ≠ target_i）不受影响，Q_i 仍
        正常接收大奖励的梯度信号。
        """
        states, actions, rewards, next_states, dones = batch
        B, N = states.shape

        s = torch.as_tensor(states, dtype=torch.long, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.long, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # post-terminal 掩码：s_i == target_i → agent 已在终点，Q_i 贡献清零
        at_target = torch.zeros(B, N, dtype=torch.float32, device=self.device)
        for i in range(N):
            t_idx = self.env.pos_to_index(*self.env.target_states[i])
            at_target[:, i] = (s[:, i] == t_idx).float()

        # 1) 当前 Q_i(s_i, a_i)，逐 agent 独立 forward
        q_list = []
        for i in range(N):
            target_idx = self.env.pos_to_index(*self.env.target_states[i])
            t_i = torch.full((B,), target_idx, dtype=torch.long, device=self.device)
            q_i = self.q_nets[i](s[:, i], t_i).gather(1, a[:, i:i + 1]).squeeze(1)  # (B,)
            q_list.append(q_i)
        q_stack = torch.stack(q_list, dim=1)              # (B, N)
        q_tot = (q_stack * (1.0 - at_target)).sum(dim=1)  # (B,)  post-terminal 已清零

        # 2) Target Q_tot：target Q_i(s'_i) → max_a，done agent 贡献置 0
        with torch.no_grad():
            q_next_list = []
            for i in range(N):
                target_idx = self.env.pos_to_index(*self.env.target_states[i])
                t_i = torch.full((B,), target_idx, dtype=torch.long, device=self.device)
                q_next_max = self.target_q_nets[i](ns[:, i], t_i).max(dim=1)[0]  # (B,)
                q_next_max = q_next_max * (1.0 - d[:, i])
                q_next_list.append(q_next_max)
            q_tot_target = torch.stack(q_next_list, dim=1).sum(dim=1)  # (B,)

            r_joint = r.sum(dim=1)                      # (B,)
            d_all = d.prod(dim=1)                       # 全员 done 时为 1
            y = r_joint + self.gamma * q_tot_target * (1.0 - d_all)

        loss = self.loss_fn(q_tot, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_nets.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------ target sync
    def update_target_qnet(self):
        """同步所有 agent Q_i 的 target 副本。"""
        for i in range(self.num_agents):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())

    # ------------------------------------------------------------------ 保存/加载
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "epsilon": self.epsilon,
            "optimizer": self.optimizer.state_dict(),
        }
        for i in range(self.num_agents):
            ckpt[f"qnet_{i}"] = self.q_nets[i].state_dict()
            ckpt[f"target_qnet_{i}"] = self.target_q_nets[i].state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            key = f"qnet_{i}"
            if key not in ckpt:
                raise KeyError(f"checkpoint missing '{key}'")
            self.q_nets[i].load_state_dict(ckpt[key])
            self.target_q_nets[i].load_state_dict(ckpt.get(f"target_qnet_{i}", ckpt[key]))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
