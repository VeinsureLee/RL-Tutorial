"""QMIX 算法包：N 个个体 Q_i + 单调 Mixer，CTDE。

公共接口::
    from rl_algorithms.qmix import QMIX

内部实现拆分::
    algo.py  - QMIX 类
    qnet.py  - per-agent Q_i 网络
    mixer.py - 单调混合网络（hypernetwork-based monotonic mixer）
"""
from rl_algorithms.qmix.algo import QMIX

__all__ = ["QMIX"]
