
"""
功能：环境验证。地图尺寸、起始/目标状态、禁区是否合法。
"""
from typing import Union
import numpy as np


def validate_environment_parameters(
    env_size: Union[tuple, list, np.ndarray],
    start_states: Union[list, tuple, np.ndarray],
    target_states: Union[list, tuple, np.ndarray],
    forbidden_areas: Union[list, tuple, np.ndarray],
) -> None:
    """
    验证环境参数。
    :param env_size: 环境大小 (rows, cols)
    :param start_states: 起始状态列表 [(x1, y1), (x2, y2), ...]
    :param target_states: 目标状态列表 [(x1, y1), (x2, y2), ...]
    :param forbidden_areas: 禁止区域列表
    """
    if not isinstance(env_size, (tuple, list, np.ndarray)) or len(env_size) != 2:
        raise ValueError(
            "Invalid environment size. Expected a tuple (rows, cols) with positive dimensions."
        )

    def check_states(states, name):
        if isinstance(states, (list, tuple, np.ndarray)):
            for idx, s in enumerate(states):
                if not isinstance(s, (tuple, list, np.ndarray)) or len(s) != 2:
                    raise ValueError(
                        f"Invalid {name}[{idx}]. Expected a tuple (x, y)."
                    )
                assert (
                    0 <= s[0] < env_size[0]
                ), f"{name}[{idx}][0] = {s[0]} out of range [0, {env_size[0]})"
                assert (
                    0 <= s[1] < env_size[1]
                ), f"{name}[{idx}][1] = {s[1]} out of range [0, {env_size[1]})"
        else:
            if not isinstance(states, (tuple, list, np.ndarray)) or len(states) != 2:
                raise ValueError(
                    f"Invalid {name}. Expected a tuple (x, y) or list of tuples."
                )
            assert 0 <= states[0] < env_size[0]
            assert 0 <= states[1] < env_size[1]

    check_states(start_states, "start_state")
    check_states(target_states, "target_state")

    if isinstance(forbidden_areas, (list, tuple, np.ndarray)):
        for idx, forbidden in enumerate(forbidden_areas):
            if isinstance(forbidden, (tuple, list)) and len(forbidden) == 2:
                pos, size = forbidden
                if isinstance(pos, (tuple, list, np.ndarray)) and len(pos) == 2:
                    assert (
                        0 <= pos[0] < env_size[0]
                    ), f"forbidden_areas[{idx}] position[0] out of range"
                    assert (
                        0 <= pos[1] < env_size[1]
                    ), f"forbidden_areas[{idx}] position[1] out of range"
                    assert (
                        pos[0] + size <= env_size[0]
                    ), f"forbidden_areas[{idx}] exceeds env_size[0]"
                    assert (
                        pos[1] + size <= env_size[1]
                    ), f"forbidden_areas[{idx}] exceeds env_size[1]"


__all__ = ["validate_environment_parameters"]
