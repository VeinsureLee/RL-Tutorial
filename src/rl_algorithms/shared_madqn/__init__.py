"""SharedMADQN 算法包：参数共享版 IQL（DTDE + Parameter Sharing）。

公共接口::
    from rl_algorithms.shared_madqn import SharedMADQN

内部实现拆分::
    algo.py  - SharedMADQN 类（继承 MADQN，让 N 个槽位指向同一份共享网络）
    qnet.py  - 共享 Q 网络（架构与 DQN/MADQN 一致）

注意：参数共享 ≠ CTDE。要真正中心化训练参考 QMIX。env 若开启
`randomize_on_reset=True`，本算法会自动改用 TargetReplayBuffer。
"""
from rl_algorithms.shared_madqn.algo import SharedMADQN

__all__ = ["SharedMADQN"]
