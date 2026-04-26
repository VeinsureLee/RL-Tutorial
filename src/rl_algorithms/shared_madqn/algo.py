"""
SharedMADQN：参数共享版 MADQN（IQL with Parameter Sharing）。

所有 agent **共用同一套** Q 网络 / 目标网络 / 优化器，但每 agent 仍保留独立
replay buffer。继承 MADQN 实现，把 self.q_nets / self.target_q_nets /
self.optimizers 三个列表的所有元素指向同一对象，take_action / update 逻辑无须
改动；update_target_qnet / save / load 重写以避免对共享对象重复操作。

与 Independent MADQN 的区别::
    Independent : N 个独立 Qnet，参数量 = N × single
    Shared      : 1 个共享 Qnet，参数量 = single

注意：本算法仍是 DTDE（去中心化训练 + 去中心化执行），并非 CTDE。每个 agent
的 update 仍然只用自己的 (s_i, a_i, r_i, s'_i, d_i)，没有团队回报 / mixer /
全局状态等执行时拿不到的额外信号。要升级到 CTDE，应该参考 VDN / QMIX。

设计取舍：
    - 同构 agent（动作空间 / 奖励结构一致）时，共享参数 = 隐式数据增强，样本效率高
    - 不同 agent 若应有不同技能偏好，共享会互相干扰
    - 本项目所有 agent 任务同构（到达自己的 target），适合参数共享

实现注意：
    一次 train_interval 内 trainer 会对每个 agent 的 buffer 独立采样 + 调用一次
    self.update(i, batch)，共享网络每步其实做了 N 次梯度步（Independent 则各网
    络各自 1 次）。若需要进一步对齐到"每步 1 次联合梯度步"，可在 trainer 里给
    SharedMADQN 加一条单独分支，把 N 个 buffer 的 batch 拼起来做一次反传。
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.shared_madqn.qnet import Qnet
from rl_algorithms.replay import ReplayBuffer, TargetReplayBuffer
from rl_algorithms.madqn import MADQN


def _build_qnet(env, hidden_dim: int) -> Qnet:
    """按 env 尺寸构建 SharedMADQN 的共享 Q 网络。"""
    return Qnet(
        state_num=env.n_states,
        action_dim=env.n_actions,
        rows=env.rows,
        cols=env.cols,
        embedding_dim=64,
        hidden_dim=hidden_dim,
    )


class SharedMADQN(MADQN):
    """参数共享 MADQN：所有 agent 共用一套 Q/target/optimizer，每 agent 独立 buffer。"""

    def __init__(self, env,
                 lr: float = 1e-4, gamma: float = 0.9,
                 epsilon: float = 0.5, epsilon_min: float = 0.01, epsilon_decay: float = 0.99,
                 hidden_dim: int = 128, update_freq: int = 100,
                 replay_buffer_size: int = 50000,
                 device: torch.device = torch.device("cpu")):
        # 不调用 super().__init__ —— 父类会创建 N 套独立网络
        self.env = env
        self.num_agents = env.num_agents
        self.n_actions = env.n_actions
        self.n_powers = env.n_powers
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.update_freq = update_freq

        # 单一共享网络
        shared_q = _build_qnet(env, hidden_dim).to(device)
        shared_t = _build_qnet(env, hidden_dim).to(device)
        shared_t.load_state_dict(shared_q.state_dict())
        shared_t.eval()
        shared_opt = optim.Adam(shared_q.parameters(), lr=lr)

        # 让 list 所有位置都指向同一对象，MADQN.take_action / update 无需改动
        self.q_nets = [shared_q] * self.num_agents
        self.target_q_nets = [shared_t] * self.num_agents
        self.optimizers = [shared_opt] * self.num_agents
        # buffer 选型取决于 env 是否在 reset 随机化：
        #   - 随机化：TargetReplayBuffer，每条 transition 同步记录 target_idx，
        #     避免 update 时把"当前 episode 的 target" 错套到来自旧 episode 的
        #     transition 上。
        #   - 否则：ReplayBuffer，跟 MADQN 一致。
        if getattr(env, "randomize_on_reset", False):
            self.buffers = [
                TargetReplayBuffer(replay_buffer_size, env, i)
                for i in range(self.num_agents)
            ]
            self._target_aware = True
        else:
            self.buffers = [ReplayBuffer(replay_buffer_size) for _ in range(self.num_agents)]
            self._target_aware = False

        self.loss_fn = nn.MSELoss()
        self.batch_size = 128

        # 单独保留引用，save/load 用
        self._shared_qnet = shared_q
        self._shared_target = shared_t
        self._shared_optimizer = shared_opt

    def update(self, agent_id: int, batch) -> float:
        """update 路径分两支：
        - _target_aware=False：直接走父类 MADQN.update（5-tuple，从 env 取 target）
        - _target_aware=True ：6-tuple，target 从 buffer 中 *逐条* 取出，避免随
          机化场景下用错 target。
        """
        if not self._target_aware:
            return super().update(agent_id, batch)

        states, actions, rewards, next_states, dones, target_idxs = batch
        s = torch.tensor(states, dtype=torch.long, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.long, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)
        t = torch.tensor(target_idxs, dtype=torch.long, device=self.device)

        q = self.q_nets[agent_id](s, t).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_q_nets[agent_id](ns, t).max(dim=1)[0]
            td_target = r + self.gamma * max_next_q * (1 - d)
        loss = self.loss_fn(q, td_target)
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()
        return float(loss.item())

    def update_target_qnet(self, agent_id: int = 0):
        """共享网络，忽略 agent_id，只同步一次。"""
        self._shared_target.load_state_dict(self._shared_qnet.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "qnet": self._shared_qnet.state_dict(),
            "target_qnet": self._shared_target.state_dict(),
            "optimizer": self._shared_optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._shared_qnet.load_state_dict(ckpt["qnet"])
        self._shared_target.load_state_dict(ckpt.get("target_qnet", ckpt["qnet"]))
        self._shared_optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
