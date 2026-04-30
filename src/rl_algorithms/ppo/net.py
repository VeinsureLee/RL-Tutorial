"""
PPO 专用 Actor-Critic 网络。

输入 (state_idx, target_idx, start_idx, [agent_id])，输出：
    - actor: 各动作的概率分布 logits
    - critic: 状态值 V(s)

架构：state/target/start embedding + 相对位置特征，可选 agent_id embedding，
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
                 embedding_dim: int = 64, hidden_dim: int = 128, num_agents: int = 1):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents

        self.embedding = nn.Embedding(state_num, embedding_dim)
        self.abs_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.target_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.start_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.rel_fc = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # 可选 agent_id embedding
        if num_agents > 1:
            self.agent_embedding = nn.Embedding(num_agents, embedding_dim)
            self.agent_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
            total_hidden = hidden_dim * 5  # h_abs, h_tgt, h_start, h_rel, h_agent
        else:
            self.agent_embedding = None
            self.agent_fc = None
            total_hidden = hidden_dim * 4  # h_abs, h_tgt, h_start, h_rel

        # Actor head: 输出 action logits
        self.actor_head = nn.Sequential(
            nn.Linear(total_hidden, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head: 输出 state value
        self.critic_head = nn.Sequential(
            nn.Linear(total_hidden, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_idx: torch.Tensor, target_idx: torch.Tensor,
                start_idx: torch.Tensor, agent_idx: torch.Tensor = None):
        """返回 (action_logits, state_value)。"""
        h_abs = self.abs_fc(self.embedding(state_idx))
        h_tgt = self.target_fc(self.embedding(target_idx))
        h_start = self.start_fc(self.embedding(start_idx))
        h_rel = self.rel_fc(_relative_features(state_idx, target_idx, self.rows, self.cols))

        if self.agent_embedding is not None and agent_idx is not None:
            h_agent = self.agent_fc(self.agent_embedding(agent_idx))
            h = torch.cat([h_abs, h_tgt, h_start, h_rel, h_agent], dim=1)
        else:
            h = torch.cat([h_abs, h_tgt, h_start, h_rel], dim=1)

        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def get_action(self, state_idx: torch.Tensor, target_idx: torch.Tensor,
                   start_idx: torch.Tensor, agent_idx: torch.Tensor = None):
        """采样动作，返回 (action, log_prob, value)。"""
        logits, value = self.forward(state_idx, target_idx, start_idx, agent_idx)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, state_idx: torch.Tensor, target_idx: torch.Tensor,
                         start_idx: torch.Tensor, actions: torch.Tensor,
                         agent_idx: torch.Tensor = None):
        """评估动作，返回 (log_prob, value, entropy)。"""
        logits, value = self.forward(state_idx, target_idx, start_idx, agent_idx)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy
