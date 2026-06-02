"""QMIX 的 Mixer 网络：用 hypernetwork 生成非负权重保证单调性。

输入：
    agent_qs: (B, N)    各 agent 选定动作的 Q 值
    state:    (B, S)    全局状态（这里用各 agent 观测的拼接近似）

输出：
    q_tot:    (B,)      联合 Q 值
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # hypernet：state -> 权重（保证 |w| 后非负，保证单调性）
        self.hyper_w1 = nn.Linear(state_dim, num_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bs = agent_qs.size(0)
        # 第一层
        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(bs, 1, self.embed_dim)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)
        # 第二层
        w2 = torch.abs(self.hyper_w2(state)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        return y.view(bs)
