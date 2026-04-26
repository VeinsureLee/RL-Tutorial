"""Independent MADQN 算法包：每 agent 一套独立 Q 网络 / buffer。

公共接口::
    from rl_algorithms.madqn import MADQN

内部实现拆分::
    algo.py  - MADQN 类（其它多 agent 算法的父类）
    qnet.py  - per-agent Q 网络
"""
from rl_algorithms.madqn.algo import MADQN

__all__ = ["MADQN"]
