
"""
其余函数：agent.yml 读写与场景编排。按 random_seed + num_agents 读取/生成并保存场景，
以及将场景写入 config/dynamic 下的独立 yml 文件。
读取/写入 yml 统一采用 utils.config_handler 的 load_yml / save_yml。
"""
import os
from typing import Union, Tuple, Optional
import numpy as np

from utils.config_handler import load_yml, save_yml
from utils.path_tool import get_abs_path
from config.states_generator import generate_states
from config.obstacle_generator import generate_forbidden_areas
from config.los_nlos import discretize_map


def _scenario_key(random_seed: int, num_agents: int) -> str:
    return f"seed_{random_seed}_agents_{num_agents}"


def _default_agent_yml_path() -> str:
    """config/dynamic/agent.yml 的绝对路径，与 config_handler 路径风格一致。"""
    return get_abs_path("config/dynamic/agent.yml")


def _dynamic_dir() -> str:
    """config/dynamic 目录路径。"""
    return os.path.dirname(_default_agent_yml_path())


def load_agent_yml(path: Optional[str] = None, encoding: str = "utf-8") -> dict:
    """加载 agent.yml（采用 utils.config_handler.load_yml），若文件或 scenarios 不存在则返回基础结构。"""
    path = path or _default_agent_yml_path()
    if not os.path.isfile(path):
        return {
            "description": "智能体与地图配置，由 config 模块按 random_seed 与 number_of_robots 生成并保存",
            "number_of_robots": {"value": 4, "description": "机器人数量"},
            "scenarios": {},
        }
    data = load_yml(path, encoding)
    if data is None:
        data = {}
    if "scenarios" not in data:
        data["scenarios"] = {}
    return data


def save_agent_yml(
    data: dict,
    path: Optional[str] = None,
    encoding: str = "utf-8",
) -> None:
    """写入 agent.yml（采用 utils.config_handler.save_yml）。"""
    path = path or _default_agent_yml_path()
    save_yml(path, data, encoding)


def load_scenario_from_agent_yml(
    random_seed: int,
    num_agents: int,
    agent_yml_path: Optional[str] = None,
) -> Optional[dict]:
    """
    从 agent.yml 中读取场景。key 为 seed_{seed}_agents_{n}。
    :return: 若存在则返回该场景 dict；否则返回 None。
    """
    data = load_agent_yml(agent_yml_path)
    key = _scenario_key(random_seed, num_agents)
    scenarios = data.get("scenarios") or {}
    return scenarios.get(key)


def save_scenario_to_agent_yml(
    random_seed: int,
    num_agents: int,
    scenario: dict,
    agent_yml_path: Optional[str] = None,
) -> None:
    """将场景写入 agent.yml 的 scenarios[seed_{seed}_agents_{n}]。"""
    data = load_agent_yml(agent_yml_path)
    key = _scenario_key(random_seed, num_agents)
    if "scenarios" not in data:
        data["scenarios"] = {}
    data["scenarios"][key] = scenario
    save_agent_yml(data, agent_yml_path)


def _is_scenario_complete(scenario: dict) -> bool:
    """判断已保存的场景是否包含完整区域信息，可直接使用、无需重新生成。"""
    if scenario is None or scenario.get("map_size") is None:
        return False
    return (
        scenario.get("los_nlos_grid") is not None
        or scenario.get("los_region") is not None
    )


def get_or_create_map_and_agents(
    random_seed: int,
    num_agents: int,
    map_size: Union[tuple, list, np.ndarray] = (48, 24),
    antenna_position: Union[tuple, list, np.ndarray] = (24, 12),
    num_forbidden_squares: int = 5,
    square_size_range: Tuple[int, int] = (3, 5),
    agent_yml_path: Optional[str] = None,
    force_regenerate: bool = False,
) -> dict:
    """
    根据 random_seed 与 num_agents 获取或生成地图与智能体配置。
    - 优先从 config/dynamic/agent.yml 读取已保存的场景，有则直接返回。
    - 若无对应 scenario 或 force_regenerate=True，则生成并保存后再返回。

    :return: 场景 dict，包含 map_size, antenna_position, forbidden_areas,
             start_states, target_states, obstacle_grid, los_region, nlos_region, los_nlos_grid 等。
    """
    if not force_regenerate:
        existing = load_scenario_from_agent_yml(random_seed, num_agents, agent_yml_path)
        if _is_scenario_complete(existing):
            return dict(existing)

    map_size = (int(map_size[0]), int(map_size[1]))
    antenna_position = (int(antenna_position[0]), int(antenna_position[1]))

    forbidden_areas = generate_forbidden_areas(
        map_size,
        antenna_position,
        num_forbidden_squares,
        square_size_range,
        random_seed,
    )
    start_states, target_states = generate_states(
        map_size, num_agents, forbidden_areas, random_seed
    )
    discrete = discretize_map(map_size, forbidden_areas, antenna_position)

    scenario = {
        "random_seed": random_seed,
        "num_agents": num_agents,
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
    save_scenario_to_agent_yml(random_seed, num_agents, scenario, agent_yml_path)
    return scenario


def scenario_to_numpy_objects(scenario: dict) -> dict:
    """
    将从 YAML 加载的场景中的 list 转回 numpy 与 set，供 env/map_config 使用。
    """
    out = dict(scenario)
    if "map_size" in out:
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


def write_map_info_to_dynamic(scenario: dict, dynamic_dir: Optional[str] = None) -> None:
    """
    将场景中的地图信息写入 config/dynamic 下的独立 yml 文件（采用 utils.config_handler.save_yml）。
    - start_states.yml, target_states.yml
    - los_region.yml, nlos_region.yml, los_nlos_grid.yml
    """
    dynamic_dir = dynamic_dir or _dynamic_dir()
    os.makedirs(dynamic_dir, exist_ok=True)

    for key, filename in [
        ("start_states", "start_states.yml"),
        ("target_states", "target_states.yml"),
        ("los_region", "los_region.yml"),
        ("nlos_region", "nlos_region.yml"),
        ("los_nlos_grid", "los_nlos_grid.yml"),
    ]:
        if key not in scenario or scenario[key] is None:
            continue
        path = os.path.join(dynamic_dir, filename)
        save_yml(path, scenario[key])


__all__ = [
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
