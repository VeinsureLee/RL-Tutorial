"""
QMIX 专用 individual Q 网络（Q_i）。

每个 agent 一份 Q_i，外层由 Mixer 组合成 Q_tot。架构与其它算法一致：
state/target embedding + 8 维相对位置特征 -> MLP -> 各复合动作 Q。

保留独立文件方便给 QMIX 单独尝试 RNN / GRU 输入或循环 Q（处理 POMDP 时常见
的修改），不会牵连 DQN / MADQN / SharedMADQN。
"""
import torch
import torch.nn as nn


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


class Qnet(nn.Module):
    """QMIX 中每 agent 的个体 Q_i：state/target embedding + 相对位置特征 -> MLP。"""

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
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_idx: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        h_abs = self.abs_fc(self.embedding(state_idx))
        h_tgt = self.target_fc(self.embedding(target_idx))
        h_rel = self.rel_fc(_relative_features(state_idx, target_idx, self.rows, self.cols))
        return self.head(torch.cat([h_abs, h_tgt, h_rel], dim=1))
