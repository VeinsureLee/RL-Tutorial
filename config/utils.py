
"""
地图与智能体生成逻辑、离散化及 agent.yml 读写。
包含：start/target 生成、环境参数校验、障碍物生成、LOS/NLOS 区域生成、
整图离散化，以及按 random_seed + num_agents 从 config/dynamic/agent.yml 读取或生成并保存。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union, List, Tuple, Set, Optional, Any
import numpy as np
import yaml


# ==================== 起始/目标状态生成 (原 agent_config) ====================

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


# ==================== 环境参数验证 (原 env_arguments) ====================

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


# ==================== 障碍物/禁区生成 (原 map_config) ====================

def _square_contains(square_position: Tuple[int, int], size: int, point: Tuple[int, int]) -> bool:
    """禁区方块是否包含某点（天线不能落在禁区内）。"""
    r, c = point[0], point[1]
    r0, c0 = square_position[0], square_position[1]
    return r0 <= r < r0 + size and c0 <= c < c0 + size


def generate_forbidden_areas(
    map_size: Union[tuple, list, np.ndarray],
    antenna_position: Union[tuple, list, np.ndarray],
    num_forbidden_squares: int,
    square_size_range: Tuple[int, int],
    random_seed: int,
) -> List[Tuple[Tuple[int, int], int]]:
    """
    生成不重叠、且不覆盖天线位置的方形禁区。
    :return: [((row, col), size), ...]
    """
    np.random.seed(random_seed)
    rows, cols = int(map_size[0]), int(map_size[1])
    ap = (int(antenna_position[0]), int(antenna_position[1]))
    forbidden_areas = []
    for _ in range(num_forbidden_squares):
        square_size = np.random.randint(
            square_size_range[0], square_size_range[1]
        )
        square_position = (
            np.random.randint(0, rows - square_size),
            np.random.randint(0, cols - square_size),
        )
        # 不与已有禁区重叠，且不覆盖天线
        while True:
            overlap = False
            for (pos, sz) in forbidden_areas:
                if not (
                    square_position[0] + square_size <= pos[0]
                    or pos[0] + sz <= square_position[0]
                    or square_position[1] + square_size <= pos[1]
                    or pos[1] + sz <= square_position[1]
                ):
                    overlap = True
                    break
            if _square_contains(square_position, square_size, ap):
                overlap = True
            if not overlap:
                break
            square_size = np.random.randint(
                square_size_range[0], square_size_range[1]
            )
            square_position = (
                np.random.randint(0, rows - square_size),
                np.random.randint(0, cols - square_size),
            )
        forbidden_areas.append((square_position, square_size))
    return forbidden_areas


def build_obstacle_grid(
    map_size: Union[tuple, list, np.ndarray],
    forbidden_areas: List[Tuple[Tuple[int, int], int]],
) -> np.ndarray:
    """
    根据禁区生成障碍物网格，True 表示该格点为障碍物。
    :return: 2D 布尔数组，形状为 map_size
    """
    rows, cols = int(map_size[0]), int(map_size[1])
    obstacle_grid = np.zeros((rows, cols), dtype=bool)
    for area in forbidden_areas:
        if isinstance(area, (tuple, list)) and len(area) == 2:
            pos, size = area
            x0, y0 = int(pos[0]), int(pos[1])
            s = int(size)
            for i in range(x0, min(x0 + s, rows)):
                for j in range(y0, min(y0 + s, cols)):
                    obstacle_grid[i, j] = True
    return obstacle_grid


# ==================== LOS/NLOS 区域 (原 map_config) ====================

def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham 直线算法，返回从 (x0,y0) 到 (x1,y1) 经过的格点坐标列表（含终点）。
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def compute_los_nlos_regions(
    map_size: Union[tuple, list, np.ndarray],
    forbidden_areas: List[Tuple[Tuple[int, int], int]],
    antenna_position: Union[tuple, list, np.ndarray],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    计算 LOS（视距）与 NLOS（非视距）区域。
    - LOS：从格点到天线连线未穿过任何禁区的格点。
    - NLOS：禁区内或连线被遮挡的格点。
    :return: (los_region, nlos_region)，均为 set of (x, y)
    """
    rows, cols = int(map_size[0]), int(map_size[1])
    ap = (int(antenna_position[0]), int(antenna_position[1]))
    obstacle_grid = build_obstacle_grid(map_size, forbidden_areas)

    los_region = set()
    nlos_region = set()

    for i in range(rows):
        for j in range(cols):
            if obstacle_grid[i, j]:
                nlos_region.add((i, j))
                continue
            line_cells = _bresenham_line(i, j, ap[0], ap[1])
            blocked = False
            for (x, y) in line_cells:
                if (x, y) == (i, j) or (x, y) == (ap[0], ap[1]):
                    continue
                if 0 <= x < rows and 0 <= y < cols and obstacle_grid[x, y]:
                    blocked = True
                    break
            if blocked:
                nlos_region.add((i, j))
            else:
                los_region.add((i, j))

    return los_region, nlos_region


def build_los_nlos_grid(
    map_size: Union[tuple, list, np.ndarray],
    los_region: Set[Tuple[int, int]],
    nlos_region: Set[Tuple[int, int]],
) -> np.ndarray:
    """
    将 LOS/NLOS 离散化为 2D 网格。1=LOS，0=NLOS。
    """
    rows, cols = int(map_size[0]), int(map_size[1])
    grid = np.zeros((rows, cols), dtype=np.int32)
    for (i, j) in los_region:
        if 0 <= i < rows and 0 <= j < cols:
            grid[i, j] = 1
    return grid


def get_los_nlos(
    x: int,
    y: int,
    los_nlos_grid: Optional[np.ndarray] = None,
    map_size: Optional[Union[tuple, list, np.ndarray]] = None,
    default_los_nlos_grid: Optional[np.ndarray] = None,
    default_map_size: Optional[Union[tuple, list, np.ndarray]] = None,
) -> str:
    """
    根据离散化坐标 (x, y) 返回该格点 LOS/NLOS。
    :return: 'los' 或 'nlos'
    """
    grid = los_nlos_grid if los_nlos_grid is not None else default_los_nlos_grid
    size = map_size if map_size is not None else default_map_size
    if grid is None or size is None:
        return "nlos"
    rows, cols = int(size[0]), int(size[1])
    if x < 0 or x >= rows or y < 0 or y >= cols:
        return "nlos"
    return "los" if grid[int(x), int(y)] == 1 else "nlos"


# ==================== 整图离散化与场景字典 ====================

def discretize_map(
    map_size: Union[tuple, list, np.ndarray],
    forbidden_areas: List[Tuple[Tuple[int, int], int]],
    antenna_position: Union[tuple, list, np.ndarray],
) -> dict:
    """
    将地图离散化：障碍物网格、LOS/NLOS 区域与网格。
    :return: dict 含 obstacle_grid, los_region, nlos_region, los_nlos_grid
             (los_region/nlos_region 为 list of [r,c] 便于 YAML 序列化)
    """
    los_region, nlos_region = compute_los_nlos_regions(
        map_size, forbidden_areas, antenna_position
    )
    obstacle_grid = build_obstacle_grid(map_size, forbidden_areas)
    los_nlos_grid = build_los_nlos_grid(map_size, los_region, nlos_region)
    return {
        "obstacle_grid": obstacle_grid.tolist(),
        "los_region": [list(p) for p in los_region],
        "nlos_region": [list(p) for p in nlos_region],
        "los_nlos_grid": los_nlos_grid.tolist(),
    }


def _scenario_key(random_seed: int, num_agents: int) -> str:
    return f"seed_{random_seed}_agents_{num_agents}"


def _default_agent_yml_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "dynamic", "agent.yml")


def load_agent_yml(path: Optional[str] = None, encoding: str = "utf-8") -> dict:
    """加载 agent.yml，若文件或 scenarios 不存在则返回基础结构。"""
    path = path or _default_agent_yml_path()
    if not os.path.isfile(path):
        return {
            "description": "智能体与地图配置，由 config.utils 按 random_seed 与 number_of_robots 生成并保存",
            "number_of_robots": {"value": 4, "description": "机器人数量"},
            "scenarios": {},
        }
    with open(path, "r", encoding=encoding) as f:
        data = yaml.safe_load(f)
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
    """写入 agent.yml。"""
    path = path or _default_agent_yml_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def load_scenario_from_agent_yml(
    random_seed: int,
    num_agents: int,
    agent_yml_path: Optional[str] = None,
) -> Optional[dict]:
    """
    从 agent.yml 中读取场景。key 为 seed_{seed}_agents_{n}。
    :return: 若存在则返回该场景 dict（含 map_size, forbidden_areas, start_states, target_states,
             los_region, nlos_region, los_nlos_grid, obstacle_grid 等）；否则返回 None。
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
    # 必须包含区域数据才视为完整（避免旧格式或残缺缓存）
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
    - 优先从 config/dynamic/agent.yml 读取已保存的场景（含区域信息），有则直接返回，不重复生成。
    - 若无对应 scenario 或 force_regenerate=True，则生成禁区、起点/终点、离散化地图与 LOS/NLOS 区域并保存后再返回。

    :return: 场景 dict，包含：
        map_size, antenna_position, forbidden_areas,
        start_states, target_states,
        obstacle_grid, los_region, nlos_region, los_nlos_grid,
        random_seed, num_agents
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
    "_default_agent_yml_path",
    "_scenario_key",
]
