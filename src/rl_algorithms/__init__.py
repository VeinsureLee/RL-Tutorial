"""强化学习模块：扁平化导出，仅暴露接口。

每个算法独立成包，包内拆 algo / qnet / (mixer)，便于按算法定位与单独修改：
    dqn/          : algo.py + qnet.py
    madqn/        : algo.py + qnet.py
    shared_madqn/ : algo.py + qnet.py
    qmix/         : algo.py + qnet.py + mixer.py
    vdn/          : algo.py + qnet.py

跨算法的公共组件直接放在本目录：
    replay.py  - ReplayBuffer / TargetReplayBuffer / JointReplayBuffer
    trainer.py - 统一训练循环
    tester.py  - 统一测试循环
    plot.py    - 训练曲线绘图
"""
from rl_algorithms.dqn import DQN
from rl_algorithms.madqn import MADQN
from rl_algorithms.shared_madqn import SharedMADQN
from rl_algorithms.qmix import QMIX
from rl_algorithms.vdn import VDN
from rl_algorithms.trainer import train
from rl_algorithms.tester import test
from rl_algorithms.plot import plot_training

__all__ = ["DQN", "MADQN", "SharedMADQN", "QMIX", "VDN", "train", "test", "plot_training"]
