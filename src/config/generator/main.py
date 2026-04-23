"""
场景生成器：依据 config/base/map.yml + random_seed.yml 生成 forbidden_areas、start_states、
target_states，并统一写入 config/dynamic/scenario.npz。

scenario.npz 键：
    map_size          (2,) int32     — 地图格数 (rows, cols)
    num_agents        () int32       — agent 数量
    antenna_position  (2,) int32     — 天线网格坐标
    start_states      (K, 2) int32
    target_states     (K, 2) int32
    forbidden_areas   (M, 3) int32   — 每行 (row, col, size)
"""
import os
import sys
from typing import Optional, Union, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from utils.path_tool import get_abs_path
from config.generator.forbidden_generator import generate_forbidden_areas
from config.generator.states_generator import generate_states
from config.generator.environment_validation import validate_environment_parameters

SCENARIO_FILENAME = "scenario.npz"


def _dynamic_dir() -> str:
    return get_abs_path("config/dynamic")


def _scenario_path(dynamic_dir: Optional[str] = None) -> str:
    return os.path.join(dynamic_dir or _dynamic_dir(), SCENARIO_FILENAME)


def _forbidden_to_array(forbidden_areas) -> np.ndarray:
    """[((r, c), size), ...] -> (M, 3) int32。"""
    if not forbidden_areas:
        return np.zeros((0, 3), dtype=np.int32)
    rows = []
    for item in forbidden_areas:
        pos, size = item
        rows.append([int(pos[0]), int(pos[1]), int(size)])
    return np.asarray(rows, dtype=np.int32)


def _array_to_forbidden(arr: np.ndarray):
    """(M, 3) int -> [((r, c), size), ...]（供 env 使用）。"""
    return [((int(r), int(c)), int(s)) for (r, c, s) in np.asarray(arr).reshape(-1, 3)]


def save_scenario(
    dynamic_dir: str,
    map_size: Tuple[int, int],
    num_agents: int,
    antenna_position: Tuple[int, int],
    start_states,
    target_states,
    forbidden_areas,
) -> str:
    os.makedirs(dynamic_dir, exist_ok=True)
    path = _scenario_path(dynamic_dir)
    np.savez_compressed(
        path,
        map_size=np.array(map_size, dtype=np.int32),
        num_agents=np.int32(num_agents),
        antenna_position=np.array(antenna_position, dtype=np.int32),
        start_states=np.asarray(start_states, dtype=np.int32).reshape(-1, 2),
        target_states=np.asarray(target_states, dtype=np.int32).reshape(-1, 2),
        forbidden_areas=_forbidden_to_array(forbidden_areas),
    )
    return path


def load_scenario(dynamic_dir: Optional[str] = None) -> Optional[dict]:
    """若 scenario.npz 不存在返回 None；否则返回 dict（与 env 接口一致）。"""
    path = _scenario_path(dynamic_dir)
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {
        "map_size": tuple(int(x) for x in data["map_size"]),
        "num_agents": int(data["num_agents"]),
        "antenna_position": tuple(int(x) for x in data["antenna_position"]),
        "start_states": [tuple(int(x) for x in row) for row in data["start_states"]],
        "target_states": [tuple(int(x) for x in row) for row in data["target_states"]],
        "forbidden_areas": _array_to_forbidden(data["forbidden_areas"]),
    }


def get_or_create_scenario(
    random_seed: int,
    num_agents: int,
    map_size: Union[tuple, list, np.ndarray] = (120, 60),
    antenna_position: Union[tuple, list, np.ndarray] = (60, 30),
    num_forbidden_squares: int = 5,
    square_size_range: Tuple[int, int] = (7, 12),
    dynamic_dir: Optional[str] = None,
    force_regenerate: bool = False,
) -> dict:
    """若已有 scenario.npz 且 agent 数一致且未强制重生成则直接加载，否则重建并落盘。"""
    dynamic_dir = dynamic_dir or _dynamic_dir()
    map_size = (int(map_size[0]), int(map_size[1]))
    antenna_position = (int(antenna_position[0]), int(antenna_position[1]))

    if not force_regenerate:
        existing = load_scenario(dynamic_dir)
        if existing is not None and existing["num_agents"] == num_agents \
                and tuple(existing["map_size"]) == map_size:
            return existing

    # 1) 障碍
    forbidden = generate_forbidden_areas(
        map_size, antenna_position, num_forbidden_squares, square_size_range, random_seed,
    )
    # 2) 起终点
    start_states, target_states = generate_states(
        map_size, num_agents, forbidden, random_seed,
    )
    # 3) 验证
    validate_environment_parameters(map_size, start_states, target_states, forbidden)
    # 4) 落盘
    save_scenario(dynamic_dir, map_size, num_agents, antenna_position,
                  start_states, target_states, forbidden)
    return {
        "map_size": map_size,
        "num_agents": num_agents,
        "antenna_position": antenna_position,
        "start_states": start_states,
        "target_states": target_states,
        "forbidden_areas": forbidden,
    }


if __name__ == "__main__":
    from config.yml_config import get_base_map_and_seed
    base = get_base_map_and_seed()
    scenario = get_or_create_scenario(
        random_seed=base["random_seed"],
        num_agents=base["num_agents"],
        map_size=base["map_size"],
        antenna_position=base["antenna_position"],
        num_forbidden_squares=base["num_forbidden_squares"],
        square_size_range=base["square_size_range"],
        force_regenerate=True,
    )
    print(f"Scenario regenerated: agents={scenario['num_agents']}, "
          f"map={scenario['map_size']}, forbidden={len(scenario['forbidden_areas'])} squares")
    print(f"Saved to: {_scenario_path()}")
