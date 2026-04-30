"""
MAPPO 专用 Actor-Critic 网络（CTDE 版本）。

每个 agent 独立的 Actor（分散执行），但每个 agent 的 Critic 用全局信息（集中训练）。

Actor 输入：(state_idx, target_idx, start_idx, agent_idx)
Critic 输入：(all_state_idx, all_start_idx, all_target_idx, all_agent_idx)
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
    """Per-agent Actor-Critic 网络（CTDE）：Actor 局部，Critic 全局。"""

    def __init__(self, state_num: int, action_dim: int, rows: int, cols: int,
                 num_agents: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.state_num = state_num
        self.action_dim = action_dim

        # --- 共享的 Embedding 层 ---
        self.embedding = nn.Embedding(state_num, embedding_dim)
        self.agent_embedding = nn.Embedding(num_agents, embedding_dim)

        # --- Actor 主干（局部信息）---
        self.actor_abs_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.actor_target_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.actor_start_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.actor_agent_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.actor_rel_fc = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # --- Critic 主干（全局信息）---
        # 每个 agent 的特征：state_emb + target_emb + start_emb + agent_emb + rel_features
        self.critic_agent_feature_dim = hidden_dim * 4 + 8
        self.critic_global_fc = nn.Sequential(
            nn.Linear(self.critic_agent_feature_dim * num_agents, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
        )
        # Critic Head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_single_agent(self, state_idx: torch.Tensor, start_idx: torch.Tensor,
                             target_idx: torch.Tensor, agent_idx: torch.Tensor):
        """编码单个 agent 的特征（用于 Critic 全局输入）。"""
        h_s = self.actor_abs_fc(self.embedding(state_idx))  # (B, hidden_dim)
        h_start = self.actor_start_fc(self.embedding(start_idx))  # (B, hidden_dim)
        h_t = self.actor_target_fc(self.embedding(target_idx))  # (B, hidden_dim)
        h_a = self.actor_agent_fc(self.agent_embedding(agent_idx))  # (B, hidden_dim)
        h_rel = _relative_features(state_idx, target_idx, self.rows, self.cols)  # (B, 8)
        return torch.cat([h_s, h_start, h_t, h_a, h_rel], dim=-1)  # (B, hidden_dim*4+8)

    def forward_actor(self, state_idx: torch.Tensor, target_idx: torch.Tensor,
                      start_idx: torch.Tensor, agent_idx: torch.Tensor):
        """Actor 前向：只用局部信息，返回 action logits。"""
        h_abs = self.actor_abs_fc(self.embedding(state_idx))
        h_tgt = self.actor_target_fc(self.embedding(target_idx))
        h_start = self.actor_start_fc(self.embedding(start_idx))
        h_agent = self.actor_agent_fc(self.agent_embedding(agent_idx))
        h_rel = self.actor_rel_fc(_relative_features(state_idx, target_idx, self.rows, self.cols))
        h = torch.cat([h_abs, h_tgt, h_start, h_agent, h_rel], dim=1)
        return self.actor_head(h)

    def forward_critic(self, all_state_idx: torch.Tensor, all_start_idx: torch.Tensor,
                       all_target_idx: torch.Tensor, all_agent_idx: torch.Tensor):
        """Critic 前向：用全局信息，返回 state value。

        输入形状：
            all_state_idx: (B, num_agents)
            all_start_idx: (B, num_agents)
            all_target_idx: (B, num_agents)
            all_agent_idx: (B, num_agents)
        """
        B = all_state_idx.shape[0]
        agent_features = []
        for i in range(self.num_agents):
            # 编码第 i 个 agent 的特征
            feat = self._encode_single_agent(
                all_state_idx[:, i], all_start_idx[:, i],
                all_target_idx[:, i], all_agent_idx[:, i]
            )
            agent_features.append(feat)
        # 拼接所有 agent 的特征
        global_feat = torch.cat(agent_features, dim=-1)  # (B, agent_feature_dim * num_agents)
        h_global = self.critic_global_fc(global_feat)
        return self.critic_head(h_global).squeeze(-1)

    def forward(self, state_idx: torch.Tensor, target_idx: torch.Tensor,
                start_idx: torch.Tensor, agent_idx: torch.Tensor = None):
        """为了兼容旧接口（只用于测试时快速前向），不推荐使用。"""
        return self.forward_actor(state_idx, target_idx, start_idx, agent_idx), None

    def get_action(self, state_idx: torch.Tensor, target_idx: torch.Tensor,
                   start_idx: torch.Tensor, agent_idx: torch.Tensor = None):
        """采样动作：Actor 只用局部信息。"""
        logits = self.forward_actor(state_idx, target_idx, start_idx, agent_idx)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 注意：这里不能计算 value，因为 value 需要全局信息！
        # value 会在 trainer 里用全局信息单独计算
        return action, log_prob, torch.tensor([0.0], device=state_idx.device)

    def evaluate_actions(self, state_idx: torch.Tensor, target_idx: torch.Tensor,
                         start_idx: torch.Tensor, actions: torch.Tensor,
                         agent_idx: torch.Tensor = None,
                         all_state_idx: torch.Tensor = None,
                         all_start_idx: torch.Tensor = None,
                         all_target_idx: torch.Tensor = None,
                         all_agent_idx: torch.Tensor = None):
        """评估动作：Actor 用局部信息，Critic 用全局信息。"""
        # Actor 部分
        logits = self.forward_actor(state_idx, target_idx, start_idx, agent_idx)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        # Critic 部分（用全局信息）
        if all_state_idx is not None:
            value = self.forward_critic(all_state_idx, all_start_idx, all_target_idx, all_agent_idx)
        else:
            value = torch.tensor([0.0] * len(state_idx), device=state_idx.device)

        return log_prob, value, entropy
