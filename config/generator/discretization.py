
"""
离散化：将地图离散为网格，并提供 O(1) 的 state 查询（是否起点、终点、障碍、LOS 区域）。
"""
import os
from typing import Union, List, Tuple, Optional
import numpy as np

from utils.config_handler import load_yml, save_yml
from utils.path_tool import get_abs_path


class StateLookup:
    """
    离散化地图的 O(1) 查询接口。
    任意 (r, c) 可在 O(1) 时间内查询：是否起点、是否终点、是否障碍、是否 LOS。
    """
    __slots__ = ("_rows", "_cols", "_start", "_target", "_obstacle", "_los")

    def __init__(
        self,
        map_size: Union[tuple, list],
        start_grid: Union[list, np.ndarray],
        target_grid: Union[list, np.ndarray],
        obstacle_grid: Union[list, np.ndarray],
        los_grid: Union[list, np.ndarray],
    ):
        self._rows, self._cols = int(map_size[0]), int(map_size[1])
        self._start = np.asarray(start_grid, dtype=bool).reshape(self._rows, self._cols)
        self._target = np.asarray(target_grid, dtype=bool).reshape(self._rows, self._cols)
        self._obstacle = np.asarray(obstacle_grid, dtype=bool).reshape(self._rows, self._cols)
        # los_grid: 1=LOS, 0=NLOS
        los = np.asarray(los_grid, dtype=np.int32).reshape(self._rows, self._cols)
        self._los = (los == 1)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._rows and 0 <= c < self._cols

    def is_start(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and bool(self._start[r, c])

    def is_target(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and bool(self._target[r, c])

    def is_obstacle(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and bool(self._obstacle[r, c])

    def is_los(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and bool(self._los[r, c])

    @property
    def map_size(self) -> Tuple[int, int]:
        return (self._rows, self._cols)


def build_state_lookup(
    map_size: Union[tuple, list, np.ndarray],
    start_states: Union[list, tuple],
    target_states: Union[list, tuple],
    obstacle_grid: Union[list, np.ndarray],
    los_nlos_grid: Union[list, np.ndarray],
) -> StateLookup:
    """
    从离散化数据构建 O(1) 查询的 StateLookup。
    :param map_size: (rows, cols)
    :param start_states: [(r,c), ...]
    :param target_states: [(r,c), ...]
    :param obstacle_grid: 2D bool，True 表示障碍
    :param los_nlos_grid: 2D int，1=LOS，0=NLOS
    """
    rows, cols = int(map_size[0]), int(map_size[1])
    start_grid = np.zeros((rows, cols), dtype=bool)
    target_grid = np.zeros((rows, cols), dtype=bool)
    for (r, c) in start_states:
        if 0 <= r < rows and 0 <= c < cols:
            start_grid[r, c] = True
    for (r, c) in target_states:
        if 0 <= r < rows and 0 <= c < cols:
            target_grid[r, c] = True
    return StateLookup(
        map_size=(rows, cols),
        start_grid=start_grid,
        target_grid=target_grid,
        obstacle_grid=obstacle_grid,
        los_grid=los_nlos_grid,
    )


FILENAME_DISCRETE_MAP = "discrete_map.yml"


def save_state_lookup(
    dynamic_dir: str,
    state_lookup: StateLookup,
    filename: str = FILENAME_DISCRETE_MAP,
) -> None:
    """将 StateLookup 保存为 yml，便于 O(1) 加载。"""
    path = os.path.join(dynamic_dir, filename)
    data = {
        "map_size": list(state_lookup.map_size),
        "start_grid": state_lookup._start.tolist(),
        "target_grid": state_lookup._target.tolist(),
        "obstacle_grid": state_lookup._obstacle.tolist(),
        "los_grid": state_lookup._los.astype(np.int32).tolist(),
    }
    save_yml(path, data)


def load_state_lookup(
    dynamic_dir: Optional[str] = None,
    filename: str = FILENAME_DISCRETE_MAP,
) -> Optional[StateLookup]:
    """
    从 config/dynamic 下的 discrete_map.yml 加载 StateLookup，支持 O(1) 查询。
    :return: StateLookup 或 None（文件不存在时）
    """
    if dynamic_dir is None:
        dynamic_dir = get_abs_path("config/dynamic")
    path = os.path.join(dynamic_dir, filename)
    if not os.path.isfile(path):
        return None
    data = load_yml(path)
    if not data:
        return None
    return StateLookup(
        map_size=data["map_size"],
        start_grid=data["start_grid"],
        target_grid=data["target_grid"],
        obstacle_grid=data["obstacle_grid"],
        los_grid=data["los_grid"],
    )


__all__ = [
    "StateLookup",
    "build_state_lookup",
    "save_state_lookup",
    "load_state_lookup",
    "FILENAME_DISCRETE_MAP",
]
