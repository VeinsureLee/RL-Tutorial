"""经验回放缓冲区：固定容量 deque，每次随机采样一个 batch。

提供三种缓冲：
- ReplayBuffer        : 单 agent 标准 5-tuple，DQN / Independent MADQN / SharedMADQN
                        在固定 (start, target) 场景下使用。
- TargetReplayBuffer  : 单 agent 6-tuple (..., target_idx)，每条 transition 在
                        采集时刻同步记录当时的 target_idx。SharedMADQN 在 reset
                        随机化场景下使用，避免反传时拿当前 episode 的 target 套
                        到来自其它 episode 的 transition 上。
- JointReplayBuffer   : N-agent 联合 transition，QMIX 用。
"""
import collections
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class TargetReplayBuffer:
    """带 target_idx 的单 agent replay buffer。

    每次 add 时从 env.target_states[agent_id] 抓取当时的 target，与 (s,a,r,ns,d)
    一起存储。采样返回 6-tuple，update 用其中的 target 而非"当前" target。这是
    reset 随机化场景下避免目标信号串扰的必要做法。
    """

    def __init__(self, capacity: int, env, agent_id: int):
        self.buffer = collections.deque(maxlen=capacity)
        self.env = env
        self.agent_id = agent_id

    def add(self, state, action, reward, next_state, done):
        target_idx = self.env.pos_to_index(*self.env.target_states[self.agent_id])
        self.buffer.append((state, action, reward, next_state, done, int(target_idx)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, targets = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(targets, dtype=np.int64),
        )

    def __len__(self):
        return len(self.buffer)


class JointReplayBuffer:
    """
    N-agent 联合 transition 存储：每条 = (states[N], actions[N], rewards[N], next_states[N], dones[N])。
    采样返回 (B, N) 形状的 batch，供 QMIX 联合 TD 更新使用。
    """

    def __init__(self, capacity: int = 50000):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, states, actions, rewards, next_states, dones):
        """输入是长度 N 的 list/array（每项对应一个 agent）。"""
        self.buffer.append((
            np.asarray(states, dtype=np.int64),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.int64),
            np.asarray(dones, dtype=np.float32),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),        # (B, N) int
            np.stack(actions),       # (B, N) int
            np.stack(rewards),       # (B, N) float
            np.stack(next_states),   # (B, N) int
            np.stack(dones),         # (B, N) float
        )

    def __len__(self):
        return len(self.buffer)
