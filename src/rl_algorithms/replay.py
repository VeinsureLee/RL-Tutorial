"""经验回放缓冲区：固定容量 deque，每次随机采样一个 batch。"""
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
