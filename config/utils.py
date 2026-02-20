
"""
地图与智能体生成逻辑、离散化及 agent.yml 读写。
本模块为统一入口，功能函数按五类分散在以下子模块中：
  1. 状态生成         — states_generator
  2. 障碍物生成       — obstacle_generator
  3. LOS/NLOS 计算生成 — los_nlos
  4. 环境验证         — environment_validation
  5. 其余函数         — agent_yml（yml 读写与场景编排，读取采用 utils.config_handler）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. 状态生成
from config.states_generator import generate_states

# 2. 障碍物生成
from config.obstacle_generator import generate_forbidden_areas, build_obstacle_grid

# 3. LOS/NLOS 计算生成
from config.los_nlos import (
    compute_los_nlos_regions,
    build_los_nlos_grid,
    get_los_nlos,
    discretize_map,
)

# 4. 环境验证
from config.environment_validation import validate_environment_parameters

# 5. 其余函数（agent.yml 读写与场景编排）
from config.agent_yml import (
    load_agent_yml,
    save_agent_yml,
    load_scenario_from_agent_yml,
    save_scenario_to_agent_yml,
    get_or_create_map_and_agents,
    scenario_to_numpy_objects,
    write_map_info_to_dynamic,
    _default_agent_yml_path,
    _dynamic_dir,
    _scenario_key,
)


__all__ = [
    "generate_states",
    "validate_environment_parameters",
    "generate_forbidden_areas",
    "build_obstacle_grid",
    "compute_los_nlos_regions",
    "build_los_nlos_grid",
    "get_los_nlos",
    "discretize_map",
    "load_agent_yml",
    "save_agent_yml",
    "load_scenario_from_agent_yml",
    "save_scenario_to_agent_yml",
    "get_or_create_map_and_agents",
    "scenario_to_numpy_objects",
    "write_map_info_to_dynamic",
    "_default_agent_yml_path",
    "_dynamic_dir",
    "_scenario_key",
]


if __name__ == "__main__":
    # 运行此文件时：按当前生成逻辑生成各地图信息，并补全到 config/dynamic 中
    DEFAULT_SEED = 42
    DEFAULT_AGENTS = 4
    MAP_SIZE = (48, 24)
    ANTENNA_POSITION = (24, 12)
    NUM_FORBIDDEN_SQUARES = 5
    SQUARE_SIZE_RANGE = (3, 5)

    scenario = get_or_create_map_and_agents(
        random_seed=DEFAULT_SEED,
        num_agents=DEFAULT_AGENTS,
        map_size=MAP_SIZE,
        antenna_position=ANTENNA_POSITION,
        num_forbidden_squares=NUM_FORBIDDEN_SQUARES,
        square_size_range=SQUARE_SIZE_RANGE,
        force_regenerate=True,
    )
    write_map_info_to_dynamic(scenario)
    print(f"地图信息已生成并写入 config/dynamic：")
    print(f"  - agent.yml（场景 seed_{DEFAULT_SEED}_agents_{DEFAULT_AGENTS}）")
    print(f"  - start_states.yml, target_states.yml")
    print(f"  - los_region.yml, nlos_region.yml, los_nlos_grid.yml")
