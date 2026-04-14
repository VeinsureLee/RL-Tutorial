"""
强化学习算法包：DQN/MADQN 结构、训练、测试与绘图接口。

快速使用:
    from rl_algorithms import train, test

    # 训练（参数可选，默认从 config/base/rl.yml 读取）
    result = train()
    result = train(algo="madqn", lr=1e-4, num_episodes=100)

    # 测试
    result = test()
    result = test(algo="madqn", model_path="models/madqn_model.pth", max_steps=300)
"""
from rl_algorithms.structure import DQN, MADQN
from rl_algorithms.train.run import train
from rl_algorithms.test.run import test
from rl_algorithms.train import train_dqn, train_madqn
from rl_algorithms.test import render_dual, render_animation_dual
from rl_algorithms.plot import plot_dqn, plot_madqn, FIG_DIR

__all__ = [
    # 统一入口（推荐）
    "train",
    "test",
    # 底层接口
    "DQN",
    "MADQN",
    "train_dqn",
    "train_madqn",
    "render_dual",
    "render_animation_dual",
    "plot_dqn",
    "plot_madqn",
    "FIG_DIR",
]
