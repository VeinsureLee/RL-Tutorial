"""
PPO 专用 Actor-Critic 网络。

输入 (state_idx, target_idx)，输出：
    - actor: 各动作的概率分布 logits
    - critic: 状态值 V(s)

架构与 DQN Q 网络类似，分支为 state/target embedding + 相对位置特征，
分别接入 actor head 和 critic head。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _relative_features(state_idx: torch.Tensor, target_idx: torch.Tensor,
                       rows: int, cols: int) -> torch.Tensor:
    """由 state_idx 与 target_idx 批量计算 8 维相对位置特征 (B, 8)。"""
    sx = state_idx // cols
    sy = state_idx % cols
    tx = target_idx // cols
    ty = target_idx % cols
    dx_n = (tx.float() - sx.float()) / rows
    dy_n = (ty.float() - sy.float()) / cols
    abs_dx = dx_n.abs()
    abs_dy = dy_n.abs()
    l1 = abs_dx + abs_dy
    l2 = torch.sqrt(dx_n ** 2 + dy_n ** 2 + 1e-8)
    sign_dx = torch.sign(dx_n)
    sign_dy = torch.sign(dy_n)
    return torch.stack([dx_n, dy_n, abs_dx, abs_dy, l1, l2, sign_dx, sign_dy], dim=1)


class ActorCriticNet(nn.Module):
    """Actor-Critic 网络：共享主干，分头输出 action logits 和 state value。"""

    def __init__(self, state_num: int, action_dim: int, rows: int, cols: int,
                 embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.embedding = nn.Embedding(state_num, embedding_dim)
        self.abs_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.target_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.rel_fc = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Actor head: 输出 action logits
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head: 输出 state value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_idx: torch.Tensor, target_idx: torch.Tensor):
        """返回 (action_logits, state_value)。"""
        h_abs = self.abs_fc(self.embedding(state_idx))
        h_tgt = self.target_fc(self.embedding(target_idx))
        h_rel = self.rel_fc(_relative_features(state_idx, target_idx, self.rows, self.cols))
        h = torch.cat([h_abs, h_tgt, h_rel], dim=1)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def get_action(self, state_idx: torch.Tensor, target_idx: torch.Tensor):
        """采样动作，返回 (action, log_prob, value)。"""
        logits, value = self.forward(state_idx, target_idx)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, state_idx: torch.Tensor, target_idx: torch.Tensor, actions: torch.Tensor):
        """评估动作，返回 (log_prob, value, entropy)。"""
        logits, value = self.forward(state_idx, target_idx)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy
