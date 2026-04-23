"""
DQN / MADQN / JointMADQN / QMIX 算法：统一 take_action / update / save / load 接口。

DQN       : 单 agent 学习，指定 agent_id；其他 agent 训练/测试时走随机策略（在 trainer 内拼接）。
MADQN     : Independent DQN，每个 agent 独立 Q 网络、目标网络、回放缓冲。
JointMADQN: 参数共享 MADQN，所有 agent 共用一套 Q / target / optimizer；相当于
            Independent 的"硬参数共享"版本，适合同构 agent，模型参数量 = MADQN / N。
QMIX      : N 个 Qnet + 单调 mixer，联合 TD target 训练；执行时分布式（各用自己的 Q_i）。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.qnet import Qnet
from rl_algorithms.replay import ReplayBuffer, JointReplayBuffer
from rl_algorithms.mixer import Mixer


def _build_qnet(env, hidden_dim: int) -> Qnet:
    """按 env 尺寸构建 Q 网络。"""
    return Qnet(
        state_num=env.n_states,
        action_dim=env.n_actions,
        rows=env.rows,
        cols=env.cols,
        embedding_dim=64,
        hidden_dim=hidden_dim,
    )


class DQN:
    """单 agent DQN：只训练 agent_id 的策略，其它 agent 由训练循环给随机动作。"""

    def __init__(self, env, agent_id: int = 0,
                 lr: float = 1e-4, gamma: float = 0.9,
                 epsilon: float = 0.5, epsilon_min: float = 0.01, epsilon_decay: float = 0.99,
                 hidden_dim: int = 128, update_freq: int = 100,
                 replay_buffer_size: int = 50000,
                 device: torch.device = torch.device("cpu")):
        self.env = env
        self.agent_id = agent_id
        self.n_actions = env.n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.update_freq = update_freq

        self.qnet = _build_qnet(env, hidden_dim).to(device)
        self.target_qnet = _build_qnet(env, hidden_dim).to(device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = 128

    def take_action(self, state, training: bool = True) -> int:
        """epsilon-greedy。state 为单个整数 state_index。"""
        target_idx = self.env.pos_to_index(*self.env.target_states[self.agent_id])
        s = torch.tensor([state], dtype=torch.long, device=self.device)
        t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.qnet(s, t)
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(q.argmax(dim=1).item())

    def update(self, batch) -> float:
        """单步 TD(0) 更新。batch=(states, actions, rewards, next_states, dones)。"""
        states, actions, rewards, next_states, dones = batch
        s = torch.tensor(states, dtype=torch.long, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.long, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)
        target_idx = self.env.pos_to_index(*self.env.target_states[self.agent_id])
        t = torch.full((len(states),), target_idx, dtype=torch.long, device=self.device)

        q = self.qnet(s, t).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_qnet(ns, t).max(dim=1)[0]
            td_target = r + self.gamma * max_next_q * (1 - d)
        loss = self.loss_fn(q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target_qnet(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "qnet": self.qnet.state_dict(),
            "target_qnet": self.target_qnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.qnet.load_state_dict(ckpt["qnet"])
        self.target_qnet.load_state_dict(ckpt.get("target_qnet", ckpt["qnet"]))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)


class MADQN:
    """Independent DQN：每 agent 一套 Q/target/buffer/optimizer，共享 env。"""

    def __init__(self, env,
                 lr: float = 1e-4, gamma: float = 0.9,
                 epsilon: float = 0.5, epsilon_min: float = 0.01, epsilon_decay: float = 0.99,
                 hidden_dim: int = 128, update_freq: int = 100,
                 replay_buffer_size: int = 50000,
                 device: torch.device = torch.device("cpu")):
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

        self.q_nets = []
        self.target_q_nets = []
        self.optimizers = []
        self.buffers = []
        for _ in range(self.num_agents):
            q = _build_qnet(env, hidden_dim).to(device)
            t = _build_qnet(env, hidden_dim).to(device)
            t.load_state_dict(q.state_dict())
            t.eval()
            self.q_nets.append(q)
            self.target_q_nets.append(t)
            self.optimizers.append(optim.Adam(q.parameters(), lr=lr))
            self.buffers.append(ReplayBuffer(replay_buffer_size))
        self.loss_fn = nn.MSELoss()
        self.batch_size = 128

    def _decode_dir(self, action: int) -> int:
        return action // self.n_powers

    def take_action(self, states, training: bool = True):
        """
        Independent 版：每 agent 完全独立 epsilon-greedy，直接返回 Q argmax（或随机），
        不做任何 agent 间冲突规避。碰撞 / 禁区 / 越界交由 env.step 处理。
        """
        actions = []
        for i in range(self.num_agents):
            if self.env.done_flags is not None and self.env.done_flags[i]:
                actions.append(0)
                continue
            if training and np.random.random() < self.epsilon:
                actions.append(int(np.random.randint(self.n_actions)))
                continue
            target_idx = self.env.pos_to_index(*self.env.target_states[i])
            s = torch.tensor([states[i]], dtype=torch.long, device=self.device)
            t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
            with torch.no_grad():
                q = self.q_nets[i](s, t)
            actions.append(int(q.argmax(dim=1).item()))
        return actions

    def update(self, agent_id: int, batch) -> float:
        states, actions, rewards, next_states, dones = batch
        s = torch.tensor(states, dtype=torch.long, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.long, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)
        target_idx = self.env.pos_to_index(*self.env.target_states[agent_id])
        t = torch.full((len(states),), target_idx, dtype=torch.long, device=self.device)

        q = self.q_nets[agent_id](s, t).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_q_nets[agent_id](ns, t).max(dim=1)[0]
            td_target = r + self.gamma * max_next_q * (1 - d)
        loss = self.loss_fn(q, td_target)
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()
        return float(loss.item())

    def update_target_qnet(self, agent_id: int):
        self.target_q_nets[agent_id].load_state_dict(self.q_nets[agent_id].state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {"epsilon": self.epsilon}
        for i in range(self.num_agents):
            ckpt[f"qnet_{i}"] = self.q_nets[i].state_dict()
            ckpt[f"target_qnet_{i}"] = self.target_q_nets[i].state_dict()
            ckpt[f"optimizer_{i}"] = self.optimizers[i].state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.q_nets[i].load_state_dict(ckpt[f"qnet_{i}"])
            self.target_q_nets[i].load_state_dict(ckpt.get(f"target_qnet_{i}", ckpt[f"qnet_{i}"]))
            self.optimizers[i].load_state_dict(ckpt[f"optimizer_{i}"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)


class JointMADQN(MADQN):
    """
    参数共享 MADQN（Joint MADQN）：所有 agent **共用同一套 Q 网络 / 目标网络 / 优化器**，
    但每 agent 保留独立 replay buffer。

    与 Independent MADQN 的区别：
        - Independent : N 个独立 Qnet，参数量 = N × single
        - Joint       : 1 个共享 Qnet，参数量 = single

    设计取舍：
        - 同构 agent（动作空间 / 奖励结构一致）时，共享参数 = 隐式数据增强，样本效率高
        - 不同 agent 若应有不同技能偏好，共享会互相干扰
        - 本项目所有 agent 任务同构（到达自己的 target），适合参数共享

    实现思路：
        继承 MADQN，重写 __init__ 把 self.q_nets / self.target_q_nets / self.optimizers
        这三个 list 的所有元素指向**同一对象**。父类的 take_action / update 逻辑不用改。
        update_target_qnet / save / load 需要重写，避免对共享对象做 N 次相同操作。

    样例状态：本类仅提供接口骨架，暂未单独做收敛性调优，调试时留意：
        - 一次 train_interval 内 trainer 会对每个 agent 的 buffer 独立采样 + 调用一次
          self.update(i, batch)，因此共享网络每步其实做了 N 次梯度步（Independent 则
          各网络各自 1 次）。
        - 若需要进一步对齐到"每步 1 次联合梯度步"，可在 trainer 里给 JointMADQN 加一
          条单独分支，把 N 个 buffer 的 batch 拼起来做一次反传。
    """

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
        self.buffers = [ReplayBuffer(replay_buffer_size) for _ in range(self.num_agents)]

        self.loss_fn = nn.MSELoss()
        self.batch_size = 128

        # 单独保留引用，save/load 用
        self._shared_qnet = shared_q
        self._shared_target = shared_t
        self._shared_optimizer = shared_opt

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


class QMIX:
    """
    QMIX：N 个 Qnet + 单调 Mixer，联合训练。

    - 执行（take_action）：每 agent 用自己的 Q_i argmax（与 MADQN 一致，包含冲突规避）。
    - 训练（update）：从 JointReplayBuffer 采 (B, N) 形 batch，
      计算 Q_tot = mixer([Q_i(s_i, a_i)]_i, s_global)，联合 MSE(Q_tot, y) 反传，
      梯度通过 mixer 自动分配到每个 Q_i（隐式 credit assignment）。
    """

    def __init__(self, env,
                 lr: float = 1e-4, gamma: float = 0.9,
                 epsilon: float = 0.5, epsilon_min: float = 0.01, epsilon_decay: float = 0.99,
                 hidden_dim: int = 128, update_freq: int = 100,
                 replay_buffer_size: int = 50000,
                 mixer_embed_dim: int = 32,
                 device: torch.device = torch.device("cpu")):
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

        # per-agent Q 网络 + target（放 ModuleList 方便统一取 parameters）
        self.q_nets = nn.ModuleList([_build_qnet(env, hidden_dim) for _ in range(self.num_agents)]).to(device)
        self.target_q_nets = nn.ModuleList([_build_qnet(env, hidden_dim) for _ in range(self.num_agents)]).to(device)
        for i in range(self.num_agents):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
            self.target_q_nets[i].eval()

        # 全局状态维度：每 agent (row_norm, col_norm, target_row_norm, target_col_norm) = 4N
        self._state_dim = 4 * self.num_agents
        self.mixer = Mixer(self.num_agents, self._state_dim, embed_dim=mixer_embed_dim).to(device)
        self.target_mixer = Mixer(self.num_agents, self._state_dim, embed_dim=mixer_embed_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        # 单一 optimizer 覆盖所有 Q 和 mixer 参数
        params = list(self.q_nets.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = JointReplayBuffer(replay_buffer_size)
        self.batch_size = 128

    # ------------------------------------------------------------------ 工具
    def _decode_dir(self, action: int) -> int:
        return action // self.n_powers

    def _states_to_global(self, states_batch: np.ndarray) -> torch.Tensor:
        """
        states_batch: (B, N) int64，每项是 pos_to_index。
        return: (B, 4N) float32 —— 每 agent 的归一化 (row, col, target_row, target_col)。
        """
        B, N = states_batch.shape
        rows, cols = self.env.rows, self.env.cols
        out = np.zeros((B, 4 * N), dtype=np.float32)
        for i in range(N):
            state_i = states_batch[:, i]
            out[:, 4 * i + 0] = (state_i // cols).astype(np.float32) / max(rows, 1)
            out[:, 4 * i + 1] = (state_i % cols).astype(np.float32) / max(cols, 1)
            tgt_r, tgt_c = self.env.target_states[i]
            out[:, 4 * i + 2] = float(tgt_r) / max(rows, 1)
            out[:, 4 * i + 3] = float(tgt_c) / max(cols, 1)
        return torch.tensor(out, device=self.device)

    # ------------------------------------------------------------------ 动作选择
    def take_action(self, states, training: bool = True):
        """
        Decentralized epsilon-greedy + 与 MADQN 一致的冲突规避（同格占用 / 禁区回退）。
        """
        actions = []
        occupied = set()
        directions = self.env.directions
        for i in range(self.num_agents):
            if self.env.done_flags is not None and self.env.done_flags[i]:
                actions.append(0)
                continue
            target_idx = self.env.pos_to_index(*self.env.target_states[i])
            s = torch.tensor([states[i]], dtype=torch.long, device=self.device)
            t = torch.tensor([target_idx], dtype=torch.long, device=self.device)
            with torch.no_grad():
                q = self.q_nets[i](s, t)
            if training and np.random.random() < self.epsilon:
                action = int(np.random.randint(self.n_actions))
            else:
                action = int(q.argmax(dim=1).item())

            dr, dc = directions[self._decode_dir(action)]
            cur_r, cur_c = self.env.positions[i]
            new_r = int(cur_r + dr)
            new_c = int(cur_c + dc)
            if (new_r, new_c) in occupied or (new_r, new_c) in self.env.forbidden_set:
                sorted_actions = q.argsort(dim=1, descending=True).squeeze().tolist()
                if isinstance(sorted_actions, int):
                    sorted_actions = [sorted_actions]
                for alt in sorted_actions:
                    dr2, dc2 = directions[self._decode_dir(alt)]
                    nr2 = int(cur_r + dr2)
                    nc2 = int(cur_c + dc2)
                    if (nr2, nc2) not in occupied and (nr2, nc2) not in self.env.forbidden_set:
                        action, new_r, new_c = alt, nr2, nc2
                        break
            occupied.add((new_r, new_c))
            actions.append(action)
        return actions

    # ------------------------------------------------------------------ 联合更新
    def update(self, batch) -> float:
        """
        batch: tuple of (states, actions, rewards, next_states, dones)，均为 (B, N) array。
        联合 TD 更新（一次反传，梯度经 mixer 流到所有 Q_i 与 mixer 自身）。
        """
        states, actions, rewards, next_states, dones = batch
        B, N = states.shape

        s = torch.tensor(states, dtype=torch.long, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.long, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 全局状态
        s_global = self._states_to_global(states)
        ns_global = self._states_to_global(next_states)

        # 1) 当前 Q_i(s_i, a_i)
        q_list = []
        for i in range(N):
            target_idx = self.env.pos_to_index(*self.env.target_states[i])
            t_i = torch.full((B,), target_idx, dtype=torch.long, device=self.device)
            q_i = self.q_nets[i](s[:, i], t_i).gather(1, a[:, i:i + 1]).squeeze(1)  # (B,)
            q_list.append(q_i)
        q_stack = torch.stack(q_list, dim=1)                     # (B, N)
        q_tot = self.mixer(q_stack, s_global)                     # (B,)

        # 2) Target Q_tot：target Q_i^-(s'_i, argmax_a Q_i^-), 已 done 的 agent 贡献置 0
        with torch.no_grad():
            q_next_list = []
            for i in range(N):
                target_idx = self.env.pos_to_index(*self.env.target_states[i])
                t_i = torch.full((B,), target_idx, dtype=torch.long, device=self.device)
                q_next_max = self.target_q_nets[i](ns[:, i], t_i).max(dim=1)[0]  # (B,)
                q_next_max = q_next_max * (1.0 - d[:, i])
                q_next_list.append(q_next_max)
            q_next_stack = torch.stack(q_next_list, dim=1)                       # (B, N)
            q_tot_target = self.target_mixer(q_next_stack, ns_global)            # (B,)

            r_joint = r.sum(dim=1)                                                # (B,)
            d_all = d.prod(dim=1)  # episode-level done：所有 agent done 时为 1
            y = r_joint + self.gamma * q_tot_target * (1.0 - d_all)

        loss = self.loss_fn(q_tot, y)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防 Q_tot 幅度大时爆梯度
        params = list(self.q_nets.parameters()) + list(self.mixer.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------ target sync
    def update_target_qnet(self):
        """同步所有 agent Q_i 与 mixer 的 target 副本。"""
        for i in range(self.num_agents):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    # ------------------------------------------------------------------ 保存/加载
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "epsilon": self.epsilon,
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        for i in range(self.num_agents):
            ckpt[f"qnet_{i}"] = self.q_nets[i].state_dict()
            ckpt[f"target_qnet_{i}"] = self.target_q_nets[i].state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.q_nets[i].load_state_dict(ckpt[f"qnet_{i}"])
            self.target_q_nets[i].load_state_dict(ckpt.get(f"target_qnet_{i}", ckpt[f"qnet_{i}"]))
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_mixer.load_state_dict(ckpt.get("target_mixer", ckpt["mixer"]))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
