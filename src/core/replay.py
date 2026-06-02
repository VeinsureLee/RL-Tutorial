"""经验回放缓冲区。

ReplayBuffer：标准单智能体（DQN 用，MADQN 用按 agent 各持一份）
JointReplayBuffer：QMIX/VDN 用，存联合 (s, a, r, s', done)
"""
import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class JointReplayBuffer:
    """存储所有智能体联合的 (s, a, r, s', done)，shape = (B, N, *)."""

    def __init__(self, capacity: int, num_agents: int, state_dim: int):
        self.capacity = capacity
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        states: dict[int, np.ndarray],
        actions: dict[int, int],
        rewards: dict[int, float],
        next_states: dict[int, np.ndarray],
        dones: dict[int, bool],
    ) -> None:
        s = np.stack([states[i] for i in range(self.num_agents)])
        a = np.array([actions[i] for i in range(self.num_agents)], dtype=np.int64)
        r = np.array([rewards[i] for i in range(self.num_agents)], dtype=np.float32)
        s_next = np.stack([next_states[i] for i in range(self.num_agents)])
        d = np.array([dones[i] for i in range(self.num_agents)], dtype=np.float32)
        self.buffer.append((s, a, r, s_next, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.stack(a),
            np.stack(r),
            np.stack(s_next).astype(np.float32),
            np.stack(d),
        )

    def __len__(self) -> int:
        return len(self.buffer)
