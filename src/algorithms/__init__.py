"""算法注册表。新增算法只需在此添加一行，无需修改 trainer。"""
from algorithms.base import BaseAlgorithm
from algorithms.value_based.dqn import DQN

ALGORITHM_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    "dqn": DQN,
}


def build_algorithm(name: str, env, cfg: dict) -> BaseAlgorithm:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {name!r}. Available: {sorted(ALGORITHM_REGISTRY)}"
        )
    return ALGORITHM_REGISTRY[name](env, cfg)


__all__ = ["BaseAlgorithm", "ALGORITHM_REGISTRY", "build_algorithm"]
