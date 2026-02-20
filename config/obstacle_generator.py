
"""
功能：障碍物生成。生成不重叠且不覆盖天线位置的方形禁区，并构建障碍物网格。
"""
from typing import Union, List, Tuple
import numpy as np


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


__all__ = [
    "generate_forbidden_areas",
    "build_obstacle_grid",
]
