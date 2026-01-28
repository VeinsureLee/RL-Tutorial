import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_algorithms.utils.utils import calculate_distance


class Qnet(nn.Module):
    """
    Q(s, a | target)

    target 通过 embedding 输入网络
    相对位置信息作为主干
    """

    def __init__(
        self,
        state_num: int,
        action_dim: int,
        x_dim: int,
        y_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        # ===== state embedding =====
        self.embedding = nn.Embedding(state_num, embedding_dim)

        self.abs_fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
        )

        # ===== target embedding branch =====
        self.target_fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
        )

        # ===== relative position branch（主干）=====
        self.rel_fc = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ===== fusion head =====
        # 融合 state embedding, target embedding 和 relative position (64 + 64 + 64 = 192)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_idx: torch.Tensor, target_idx: torch.Tensor):
        """
        state_idx: (B,)
        target_idx: (B,)
        """

        # ===== absolute state embedding =====
        state_emb = self.embedding(state_idx)
        h_abs = self.abs_fc(state_emb)

        # ===== target embedding =====
        target_emb = self.embedding(target_idx)
        h_target = self.target_fc(target_emb)

        # ===== relative position to target using calculate_distance =====
        # 使用 calculate_distance 计算相对位置特征（批量处理）
        device = state_idx.device
        distance_info = calculate_distance(
            state_idx,
            target_idx,
            y_dim=self.y_dim,
            x_dim=self.x_dim,
            device=device
        )

        # 提取相对位置特征并堆叠 (B, 8)
        rel_feat = torch.stack([
            distance_info['dx_n'],
            distance_info['dy_n'],
            distance_info['abs_dx'],
            distance_info['abs_dy'],
            distance_info['l1'],
            distance_info['l2'],
            distance_info['sign_dx'],
            distance_info['sign_dy'],
        ], dim=1)  # (B, 8)

        h_rel = self.rel_fc(rel_feat)

        # ===== fusion =====
        h = torch.cat([h_abs, h_target, h_rel], dim=1)
        q = self.head(h)

        return q
