"""VDN 算法包：N 个个体 Q_i 加法求和得 Q_tot，CTDE。

公共接口::
    from rl_algorithms.vdn import VDN

内部实现拆分::
    algo.py  - VDN 类
    qnet.py  - per-agent Q_i 网络
"""
from rl_algorithms.vdn.algo import VDN

__all__ = ["VDN"]
