
"""
功能：LOS/NLOS 计算生成。视距/非视距区域、整图离散化（障碍物网格 + LOS/NLOS 网格）。
"""
from typing import Union, List, Tuple, Set, Optional
import numpy as np

from config.generator.forbidden_generator import build_obstacle_grid


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


__all__ = [
    "compute_los_nlos_regions",
    "build_los_nlos_grid",
    "get_los_nlos",
    "discretize_map",
]
