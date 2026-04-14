"""
训练子模块：DQN/MADQN 训练逻辑。
入口为 train()，可从 rl_algorithms.train 调用或 python -m rl_algorithms.train。
"""
from rl_algorithms.train.train_dqn import train_dqn
from rl_algorithms.train.train_madqn import train_madqn
from rl_algorithms.train.run import train, main

__all__ = ["train", "train_dqn", "train_madqn", "main"]
