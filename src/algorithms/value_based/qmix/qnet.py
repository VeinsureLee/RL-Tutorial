"""QMIX 的 agent 个体 Q 网络。"""
import torch
import torch.nn as nn


class AgentQNet(nn.Module):
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
