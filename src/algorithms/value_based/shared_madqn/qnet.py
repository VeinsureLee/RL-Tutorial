"""SharedMADQN 的共享 Q 网络（所有 agent 共用一份参数）。"""
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
