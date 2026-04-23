"""
QMIX 单调混合网络（mixer）。

论文：Rashid et al. 2018, "QMIX: Monotonic Value Function Factorisation for Deep
Multi-Agent Reinforcement Learning"。

给定 N 个 agent 的 Q_i(s_i, a_i) 以及全局状态 s，mixer 输出联合 Q_tot：

    Q_tot = w2 · ELU(W1 @ [Q_1, ..., Q_N] + b1) + b2

其中 W1, w2 由 hypernetwork(s) 生成，并对生成结果 abs(·) 以保证 **单调性**（
∂Q_tot/∂Q_i ≥ 0）。单调性保证 argmax_{a_i} Q_i 联合起来等价于 argmax Q_tot，
因此训练时中心化、执行时分布式。

b2 来自两层 MLP hypernetwork，不做非负约束（相当于 state-dependent value bias）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mixer(nn.Module):
    """QMIX mixer：Q_tot = f(Q_vec; state) 的单调 MLP。"""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32):
        """
        :param num_agents: agent 数量 N
        :param state_dim:  全局状态维度（一般是 N 个 agent 位置/目标的拼接）
        :param embed_dim:  mixer 隐层宽度
        """
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # 第一层：W1 ∈ R^{N × E}
        self.hyper_w1 = nn.Linear(state_dim, num_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        # 第二层：W2 ∈ R^{E × 1}
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        # 终偏置 b2：两层 MLP 输出标量（不做单调约束）
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        :param agent_qs: (B, N) 每个 agent 的 Q_i(s_i, a_i)
        :param states:   (B, state_dim) 全局状态
        :return: (B,) 联合 Q_tot
        """
        B = agent_qs.shape[0]
        # 第一层权重与偏置：单调性靠 abs 保证
        w1 = torch.abs(self.hyper_w1(states)).view(B, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)
        # (B, 1, N) @ (B, N, E) -> (B, 1, E)
        h = torch.bmm(agent_qs.view(B, 1, self.num_agents), w1) + b1
        h = F.elu(h)

        # 第二层
        w2 = torch.abs(self.hyper_w2(states)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(B, 1, 1)
        # (B, 1, E) @ (B, E, 1) -> (B, 1, 1)
        q_tot = torch.bmm(h, w2) + b2
        return q_tot.view(B)
