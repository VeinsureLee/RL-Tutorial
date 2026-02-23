"""
地图与智能体生成：起点/终点、障碍区、LOS/NLOS 区域、环境验证、离散化与 O(1) 查询。
外部按需从本包导入：get_or_create_map_and_agents、get_state_lookup、load_scenario_from_dynamic 等。
"""
from config.generator.main import (
    get_or_create_map_and_agents,
    get_state_lookup,
    load_scenario_from_dynamic,
    scenario_to_numpy_objects,
    FILENAME_AGENT_NUM,
)
from config.generator.discretization import (
    StateLookup,
    build_state_lookup,
    save_state_lookup,
    load_state_lookup,
    FILENAME_DISCRETE_MAP,
)
from config.generator.states_generator import generate_states
from config.generator.forbidden_generator import generate_forbidden_areas, build_obstacle_grid
from config.generator.region_generator import (
    compute_los_nlos_regions,
    build_los_nlos_grid,
    get_los_nlos,
    discretize_map,
)
from config.generator.environment_validation import validate_environment_parameters

__all__ = [
    "get_or_create_map_and_agents",
    "get_state_lookup",
    "load_scenario_from_dynamic",
    "scenario_to_numpy_objects",
    "FILENAME_AGENT_NUM",
    "StateLookup",
    "build_state_lookup",
    "save_state_lookup",
    "load_state_lookup",
    "FILENAME_DISCRETE_MAP",
    "generate_states",
    "generate_forbidden_areas",
    "build_obstacle_grid",
    "compute_los_nlos_regions",
    "build_los_nlos_grid",
    "get_los_nlos",
    "discretize_map",
    "validate_environment_parameters",
]
