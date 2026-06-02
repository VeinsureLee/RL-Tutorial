"""算法注册表。新增算法只需在此添加一行，无需修改 trainer。"""
from algorithms.base import BaseAlgorithm
from algorithms.policy_based.mappo import MAPPO
from algorithms.policy_based.ppo import PPO
from algorithms.value_based.dqn import DQN
from algorithms.value_based.madqn import MADQN
from algorithms.value_based.qmix import QMIX
from algorithms.value_based.shared_madqn import SharedMADQN
from algorithms.value_based.vdn import VDN

ALGORITHM_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    # value-based
    "dqn": DQN,
    "madqn": MADQN,
    "shared_madqn": SharedMADQN,
    "vdn": VDN,
    "qmix": QMIX,
    # policy-based
    "ppo": PPO,
    "mappo": MAPPO,
}


def build_algorithm(name: str, env, cfg: dict) -> BaseAlgorithm:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {name!r}. Available: {sorted(ALGORITHM_REGISTRY)}"
        )
    return ALGORITHM_REGISTRY[name](env, cfg)


__all__ = ["BaseAlgorithm", "ALGORITHM_REGISTRY", "build_algorithm"]
