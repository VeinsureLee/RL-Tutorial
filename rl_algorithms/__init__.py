"""
强化学习算法包：DQN/MADQN 结构、训练、测试与绘图接口。
"""
from rl_algorithms.structure import DQN, MADQN
from rl_algorithms.train import train_dqn, train_madqn, main
from rl_algorithms.test import render_dual, render_animation_dual
from rl_algorithms.plot import plot_dqn, plot_madqn, FIG_DIR

__all__ = [
    "DQN",
    "MADQN",
    "train_dqn",
    "train_madqn",
    "main",
    "render_dual",
    "render_animation_dual",
    "plot_dqn",
    "plot_madqn",
    "FIG_DIR",
]
