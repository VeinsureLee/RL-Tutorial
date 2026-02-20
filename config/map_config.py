
"""
地图设置与禁区、LOS/NLOS 由 config.utils 根据 random_seed 与 number_of_robots
从 config/dynamic/agent.yml 获取或生成并保存。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union
import numpy as np
from config.param_arguments import parser
from config import utils as config_utils


args = parser.parse_args()
antenna_position = args.antenna_position
num_agents = args.number_of_robots


# ==================== Map settings ====================
map_size: Union[list, tuple, np.ndarray] = (48, 24)
# ==================== End of map settings ====================

# ==================== Forbidden areas / LOS-NLOS：优先从 agent.yml 读取已保存区域，无则生成并保存 ====================
RANDOM_SEED = 42
NUM_FORBIDDEN_SQUARES = 5
SQUARE_SIZE_RANGE = (3, 5)

_scenario = config_utils.get_or_create_map_and_agents(
    random_seed=RANDOM_SEED,
    num_agents=num_agents,
    map_size=map_size,
    antenna_position=antenna_position,
    num_forbidden_squares=NUM_FORBIDDEN_SQUARES,
    square_size_range=SQUARE_SIZE_RANGE,
    agent_yml_path=config_utils._default_agent_yml_path(),
)
_scenario = config_utils.scenario_to_numpy_objects(_scenario)

forbidden_areas = _scenario["forbidden_areas"]
start_states = _scenario["start_states"]
target_states = _scenario["target_states"]
los_region = _scenario["los_region"]
nlos_region = _scenario["nlos_region"]
los_nlos_grid = _scenario["los_nlos_grid"]


def get_los_nlos(x: int, y: int, los_nlos_grid_=None, map_size_=None):
    """
    根据离散化后的坐标 (x, y) 返回该格点的 LOS/NLOS 分类。
    """
    return config_utils.get_los_nlos(
        x, y,
        los_nlos_grid=los_nlos_grid_,
        map_size=map_size_,
        default_los_nlos_grid=los_nlos_grid,
        default_map_size=map_size,
    )


# 保留对外接口名，供其他模块调用
compute_los_nlos_regions = config_utils.compute_los_nlos_regions
build_los_nlos_grid = config_utils.build_los_nlos_grid


__all__ = [
    "map_size", "forbidden_areas", "start_states", "target_states",
    "RANDOM_SEED", "SQUARE_SIZE_RANGE", "NUM_FORBIDDEN_SQUARES",
    "compute_los_nlos_regions", "los_region", "nlos_region",
    "build_los_nlos_grid", "los_nlos_grid", "get_los_nlos",
]
