
"""
生成器入口：从 config 读取参数，依次调用
1) 障碍区生成 2) 起点终点生成 3) LOS/NLOS 区域生成 4) 环境验证 5) 离散化，
并写入 config/dynamic 的 yml 及离散化后的 O(1) 查询文件。
"""
import sys
import os
from typing import Union, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from utils.config_handler import load_yml, save_yml
from utils.path_tool import get_abs_path
from config.generator.forbidden_generator import generate_forbidden_areas
from config.generator.states_generator import generate_states
from config.generator.region_generator import (
    compute_los_nlos_regions,
    build_obstacle_grid,
    build_los_nlos_grid,
    discretize_map,
)
from config.generator.environment_validation import validate_environment_parameters
from config.generator.discretization import (
    build_state_lookup,
    save_state_lookup,
    load_state_lookup,
    StateLookup,
    FILENAME_DISCRETE_MAP,
)

FILENAME_AGENT_NUM = "agent_num.yml"


def _dynamic_dir() -> str:
    return get_abs_path("config/dynamic")


def _file_path(dynamic_dir: str, filename: str) -> str:
    return os.path.join(dynamic_dir, filename)


def _to_serializable(val):
    if val is None:
        return None
    if hasattr(val, "__iter__") and not isinstance(val, (str, dict)):
        if isinstance(val, set):
            return [list(p) for p in val]
        if isinstance(val, np.ndarray):
            return val.tolist()
    return val


def _write_yml_files(
    dynamic_dir: str,
    scenario: dict,
) -> None:
    """将场景写入 start_states, target_states, forbidden_areas, los_region, nlos_region, los_nlos_grid。"""
    os.makedirs(dynamic_dir, exist_ok=True)
    for key, filename in [
        ("start_states", "start_states.yml"),
        ("target_states", "target_states.yml"),
        ("forbidden_areas", "forbidden_areas.yml"),
        ("los_region", "los_region.yml"),
        ("nlos_region", "nlos_region.yml"),
        ("los_nlos_grid", "los_nlos_grid.yml"),
    ]:
        if key not in scenario or scenario[key] is None:
            continue
        save_yml(_file_path(dynamic_dir, filename), _to_serializable(scenario[key]))


def _write_agent_num(dynamic_dir: str, num_agents: int) -> None:
    """将 agent 数量暂存到 config/dynamic/agent_num.yml。"""
    save_yml(_file_path(dynamic_dir, FILENAME_AGENT_NUM), {"num_agents": num_agents})


def _is_dynamic_complete(dynamic_dir: str) -> bool:
    required = [
        "start_states.yml", "target_states.yml", "forbidden_areas.yml",
        "los_region.yml", "nlos_region.yml", "los_nlos_grid.yml",
        FILENAME_DISCRETE_MAP,
    ]
    return all(os.path.isfile(_file_path(dynamic_dir, f)) for f in required)


def load_scenario_from_dynamic(
    dynamic_dir: Optional[str] = None,
    map_size: Optional[Union[tuple, list, np.ndarray]] = None,
    encoding: str = "utf-8",
) -> Optional[dict]:
    """
    从 config/dynamic 下各 yml 加载场景。
    若任一必需文件不存在则返回 None。
    """
    dynamic_dir = dynamic_dir or _dynamic_dir()
    required = [
        "start_states.yml", "target_states.yml", "forbidden_areas.yml",
        "los_region.yml", "nlos_region.yml", "los_nlos_grid.yml",
    ]
    for f in required:
        if not os.path.isfile(_file_path(dynamic_dir, f)):
            return None

    def load(name: str):
        return load_yml(_file_path(dynamic_dir, name), encoding)

    start_raw = load("start_states.yml")
    target_raw = load("target_states.yml")
    forbidden_raw = load("forbidden_areas.yml")
    los_raw = load("los_region.yml")
    nlos_raw = load("nlos_region.yml")
    grid_raw = load("los_nlos_grid.yml")

    if any(x is None for x in (start_raw, target_raw, forbidden_raw, los_raw, nlos_raw, grid_raw)):
        return None

    scenario = {
        "map_size": None,
        "forbidden_areas": forbidden_raw,
        "start_states": start_raw,
        "target_states": target_raw,
        "los_region": los_raw,
        "nlos_region": nlos_raw,
        "los_nlos_grid": grid_raw,
    }
    if map_size is not None:
        scenario["map_size"] = np.array(map_size)
    elif isinstance(grid_raw, (list, np.ndarray)):
        grid = np.array(grid_raw)
        if grid.ndim == 2:
            scenario["map_size"] = np.array([grid.shape[0], grid.shape[1]])
    return scenario_to_numpy_objects(scenario)


def scenario_to_numpy_objects(scenario: dict) -> dict:
    """将 YAML 加载的 list 转回 numpy 与 set，供 env/yml_config 等使用。"""
    out = dict(scenario)
    if "map_size" in out and out["map_size"] is not None:
        out["map_size"] = np.array(out["map_size"])
    if "forbidden_areas" in out:
        out["forbidden_areas"] = [
            (tuple(pos), int(sz)) for pos, sz in out["forbidden_areas"]
        ]
    if "start_states" in out:
        out["start_states"] = [tuple(s) for s in out["start_states"]]
    if "target_states" in out:
        out["target_states"] = [tuple(t) for t in out["target_states"]]
    if "obstacle_grid" in out:
        out["obstacle_grid"] = np.array(out["obstacle_grid"], dtype=bool)
    if "los_region" in out:
        out["los_region"] = set(tuple(p) for p in out["los_region"])
    if "nlos_region" in out:
        out["nlos_region"] = set(tuple(p) for p in out["nlos_region"])
    if "los_nlos_grid" in out:
        out["los_nlos_grid"] = np.array(out["los_nlos_grid"], dtype=np.int32)
    return out


def get_or_create_map_and_agents(
    random_seed: int,
    num_agents: int,
    map_size: Union[tuple, list, np.ndarray] = (48, 24),
    antenna_position: Union[tuple, list, np.ndarray] = (24, 12),
    num_forbidden_squares: int = 5,
    square_size_range: Tuple[int, int] = (3, 5),
    dynamic_dir: Optional[str] = None,
    force_regenerate: bool = False,
) -> dict:
    """
    若 config/dynamic 下已有完整 yml 且未强制重新生成则加载；否则按顺序：
    1) 障碍区 2) 起点终点 3) LOS/NLOS 4) 环境验证 5) 离散化并保存 yml + O(1) 查询文件。
    """
    dynamic_dir = dynamic_dir or _dynamic_dir()
    map_size = (int(map_size[0]), int(map_size[1]))

    if not force_regenerate and _is_dynamic_complete(dynamic_dir):
        loaded = load_scenario_from_dynamic(dynamic_dir=dynamic_dir, map_size=map_size)
        if loaded is not None and loaded.get("map_size") is not None:
            return loaded

    antenna_position = (int(antenna_position[0]), int(antenna_position[1]))

    # 1) 障碍区
    forbidden_areas = generate_forbidden_areas(
        map_size, antenna_position, num_forbidden_squares, square_size_range, random_seed,
    )
    # 2) 起点、终点
    start_states, target_states = generate_states(
        map_size, num_agents, forbidden_areas, random_seed,
    )
    # 3) LOS/NLOS 区域与离散化数据
    discrete = discretize_map(map_size, forbidden_areas, antenna_position)
    los_nlos_grid = np.array(discrete["los_nlos_grid"], dtype=np.int32)
    obstacle_grid = np.array(discrete["obstacle_grid"], dtype=bool)

    # 4) 环境验证
    validate_environment_parameters(
        map_size, start_states, target_states, forbidden_areas,
    )

    scenario = {
        "map_size": list(map_size),
        "antenna_position": list(antenna_position),
        "forbidden_areas": [[list(pos), size] for (pos, size) in forbidden_areas],
        "start_states": [list(s) for s in start_states],
        "target_states": [list(t) for t in target_states],
        "obstacle_grid": discrete["obstacle_grid"],
        "los_region": discrete["los_region"],
        "nlos_region": discrete["nlos_region"],
        "los_nlos_grid": discrete["los_nlos_grid"],
    }
    _write_yml_files(dynamic_dir, scenario)
    _write_agent_num(dynamic_dir, num_agents)

    # 5) 离散化 O(1) 查询：构建并保存 StateLookup
    state_lookup = build_state_lookup(
        map_size, start_states, target_states, obstacle_grid, los_nlos_grid,
    )
    save_state_lookup(dynamic_dir, state_lookup)

    return scenario_to_numpy_objects({
        **scenario,
        "map_size": np.array(map_size),
        "forbidden_areas": forbidden_areas,
        "start_states": start_states,
        "target_states": target_states,
        "obstacle_grid": np.array(obstacle_grid),
        "los_region": set(tuple(p) for p in discrete["los_region"]),
        "nlos_region": set(tuple(p) for p in discrete["nlos_region"]),
        "los_nlos_grid": los_nlos_grid,
    })


def get_state_lookup(dynamic_dir: Optional[str] = None) -> Optional[StateLookup]:
    """
    加载离散化后的 O(1) 查询对象。
    任意 state (r, c) 可 O(1) 查询：is_start(r,c), is_target(r,c), is_obstacle(r,c), is_los(r,c)。
    """
    return load_state_lookup(dynamic_dir=dynamic_dir)


if __name__ == "__main__":
    from config.yml_config import get_base_map_and_seed
    base = get_base_map_and_seed()
    scenario = get_or_create_map_and_agents(
        random_seed=base["random_seed"],
        num_agents=base["num_agents"],
        map_size=base["map_size"],
        antenna_position=base["antenna_position"],
        num_forbidden_squares=base["num_forbidden_squares"],
        square_size_range=base["square_size_range"],
        dynamic_dir=_dynamic_dir(),
        force_regenerate=True,
    )
    print("地图信息已生成并写入 config/dynamic：")
    print("  - start_states.yml, target_states.yml, forbidden_areas.yml")
    print("  - los_region.yml, nlos_region.yml, los_nlos_grid.yml")
    print("  - discrete_map.yml（离散化，O(1) 查询起点/终点/障碍/LOS）")
    lookup = get_state_lookup()
    if lookup:
        r, c = scenario["start_states"][0]
        print(f"  示例 O(1) 查询 state ({r},{c}): is_start={lookup.is_start(r,c)}, is_los={lookup.is_los(r,c)}")
