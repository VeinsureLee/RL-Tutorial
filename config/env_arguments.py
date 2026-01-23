__credits__ = ["Graduation project at BUPT."]
'''
Specify parameters of the environment
'''
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union
import numpy as np
import argparse
from config.map_config import map_size, forbidden_areas
from config.agent_config import start_states, target_states


env_parser = argparse.ArgumentParser("Settings for Environment")


## ==================== User settings ====================
# specify the size of the environment
env_parser.add_argument("--map-size", type=Union[list, tuple, np.ndarray],
                        default=map_size, )

# specify the size of the grid
env_parser.add_argument("--grid_size", type=float, default=0.4)

# specify the start state
env_parser.add_argument("--start-states", type=Union[list, tuple, np.ndarray],
                        default=start_states)

# specify the target state
env_parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray],
                        default=target_states)

# specify the forbidden states
env_parser.add_argument("--forbidden-areas", type=eval,
                        default=forbidden_areas)

# specify the reward when reaching target
env_parser.add_argument("--reward-target", type=float,
                        default=10)

# specify the reward when entering into forbidden area
env_parser.add_argument("--reward-forbidden", type=float,
                        default=-5)

# specify the reward for each step
env_parser.add_argument("--reward-step", type=float,
                        default=-1)
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
env_parser.add_argument("--action-space", type=Union[list, tuple, np.ndarray],
                        default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)])  # down, right, up, left, stay
env_parser.add_argument("--debug", type=bool,
                        default=False)
env_parser.add_argument("--animation-interval", type=float,
                        default=0.2)
## ==================== End of Advanced settings ====================


args = env_parser.parse_args()


def validate_environment_parameters(env_size, start_states, target_states, forbidden_areas):
    """
    验证环境参数
    :param env_size: 环境大小 (rows, cols)
    :param start_states: 起始状态列表 [(x1, y1), (x2, y2), ...]
    :param target_states: 目标状态列表 [(x1, y1), (x2, y2), ...]
    :param forbidden_areas: 禁止区域列表
    """
    if not (isinstance(env_size, (tuple, list, np.ndarray))) or len(env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    # 验证所有起始状态
    if isinstance(start_states, (list, tuple, np.ndarray)):
        for idx, start_state in enumerate(start_states):
            if not isinstance(start_state, (tuple, list, np.ndarray)) or len(start_state) != 2:
                raise ValueError(f"Invalid start_state[{idx}]. Expected a tuple (x, y).")
            assert 0 <= start_state[0] < env_size[0], f"start_state[{idx}][0] = {start_state[0]} out of range [0, {env_size[0]})"
            assert 0 <= start_state[1] < env_size[1], f"start_state[{idx}][1] = {start_state[1]} out of range [0, {env_size[1]})"
    else:
        # 单个状态的情况（向后兼容）
        if not isinstance(start_states, (tuple, list, np.ndarray)) or len(start_states) != 2:
            raise ValueError("Invalid start_state. Expected a tuple (x, y) or list of tuples.")
        assert 0 <= start_states[0] < env_size[0]
        assert 0 <= start_states[1] < env_size[1]
    
    # 验证所有目标状态
    if isinstance(target_states, (list, tuple, np.ndarray)):
        for idx, target_state in enumerate(target_states):
            if not isinstance(target_state, (tuple, list, np.ndarray)) or len(target_state) != 2:
                raise ValueError(f"Invalid target_state[{idx}]. Expected a tuple (x, y).")
            assert 0 <= target_state[0] < env_size[0], f"target_state[{idx}][0] = {target_state[0]} out of range [0, {env_size[0]})"
            assert 0 <= target_state[1] < env_size[1], f"target_state[{idx}][1] = {target_state[1]} out of range [0, {env_size[1]})"
    else:
        # 单个状态的情况（向后兼容）
        if not isinstance(target_states, (tuple, list, np.ndarray)) or len(target_states) != 2:
            raise ValueError("Invalid target_state. Expected a tuple (x, y) or list of tuples.")
        assert 0 <= target_states[0] < env_size[0]
        assert 0 <= target_states[1] < env_size[1]
    
    # 验证禁止区域
    if isinstance(forbidden_areas, (list, tuple, np.ndarray)):
        for idx, forbidden in enumerate(forbidden_areas):
            if isinstance(forbidden, (tuple, list)) and len(forbidden) == 2:
                # 格式: ((x, y), size)
                pos, size = forbidden
                if isinstance(pos, (tuple, list, np.ndarray)) and len(pos) == 2:
                    assert 0 <= pos[0] < env_size[0], f"forbidden_areas[{idx}] position[0] = {pos[0]} out of range"
                    assert 0 <= pos[1] < env_size[1], f"forbidden_areas[{idx}] position[1] = {pos[1]} out of range"
                    assert pos[0] + size <= env_size[0], f"forbidden_areas[{idx}] exceeds env_size[0]"
                    assert pos[1] + size <= env_size[1], f"forbidden_areas[{idx}] exceeds env_size[1]"


try:
    validate_environment_parameters(args.map_size, args.start_states, args.target_state, args.forbidden_areas)
except (ValueError, AssertionError) as e:
    print("Error:", e)