
"""
功能：状态生成。在给定地图与禁区下生成不重叠的 start_states 与 target_states。
"""
from typing import Union, List, Tuple
import numpy as np


def generate_states(
    map_size: Union[tuple, list, np.ndarray],
    num_agents: int,
    forbidden_areas: List[Tuple[Tuple[int, int], int]],
    random_seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    生成不重叠且不在禁区内的 start_states 与 target_states。
    :param map_size: (rows, cols)
    :param num_agents: 智能体数量
    :param forbidden_areas: [((row, col), size), ...]
    :param random_seed: 随机种子
    :return: (start_states, target_states)，每个为 [(r,c), ...]
    """
    np.random.seed(random_seed)
    rows, cols = int(map_size[0]), int(map_size[1])
    start_states = []
    target_states = []

    occupied_positions = set()
    for forbidden in forbidden_areas:
        pos, size = forbidden
        for r in range(pos[0], pos[0] + size):
            for c in range(pos[1], pos[1] + size):
                occupied_positions.add((r, c))

    def is_valid(pos):
        return (
            pos not in occupied_positions
            and 0 <= pos[0] < rows
            and 0 <= pos[1] < cols
        )

    while len(start_states) < num_agents:
        candidate = (np.random.randint(0, rows), np.random.randint(0, cols))
        if is_valid(candidate) and candidate not in start_states:
            start_states.append(candidate)
            occupied_positions.add(candidate)

    while len(target_states) < num_agents:
        candidate = (np.random.randint(0, rows), np.random.randint(0, cols))
        if (
            is_valid(candidate)
            and candidate not in target_states
            and candidate not in start_states
        ):
            target_states.append(candidate)
            occupied_positions.add(candidate)

    return start_states, target_states


__all__ = ["generate_states"]
