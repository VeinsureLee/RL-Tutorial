"""MADQN 的 Q 网络（与 DQN 相同结构，每个 agent 独立持有一份）。"""
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
