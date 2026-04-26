"""DQN 算法包：单 agent 经典 DQN。

公共接口::
    from rl_algorithms.dqn import DQN

内部实现拆分::
    algo.py  - DQN 类
    qnet.py  - Q 网络
"""
from rl_algorithms.dqn.algo import DQN

__all__ = ["DQN"]
