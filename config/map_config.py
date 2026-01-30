
"""
This file combines map settings and forbidden area generation
You can configure map parameters and random square forbidden areas
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union
import numpy as np
from config.param_arguments import parser


antenna_position = parser.parse_args().antenna_position


# ==================== Map settings ====================
# specify the size of map
map_size: Union[list, tuple, np.ndarray] = (48, 24)
# ==================== End of map settings ====================


# ==================== Forbidden areas basic settings ====================
# specify the random seed
RANDOM_SEED = 42

# specify the number of forbidden areas(squares)
NUM_FORBIDDEN_SQUARES = 5

# specify the range of square size
SQUARE_SIZE_RANGE = (3, 5)
# ==================== End of forbidden areas basic settings ====================


# ==================== Logistic of forbidden areas generation ====================
def generate_forbidden_areas(map_size, 
                             antenna_position, 
                             num_forbidden_squares, 
                             square_size_range, 
                             random_seed):
    """
        Generate forbidden areas on the map, 
        forbidden areas can't cover the antenna position, 
        and can't overlap with each other
        :param map_size: the size of map
        :param antenna_position: the position of antenna
        :param num_forbidden_squares: the number of forbidden areas(squares)
        :param square_size_range: the range of square size
        :param random_seed: the random seed
        :return: the forbidden areas
    """
    np.random.seed(random_seed)
    forbidden_areas = []
    for _ in range(num_forbidden_squares):
        square_size = np.random.randint(square_size_range[0], square_size_range[1])
        square_position = (np.random.randint(0, map_size[0] - square_size), np.random.randint(0, map_size[1] - square_size))
        while square_position in forbidden_areas or square_position in antenna_position:
            square_size = np.random.randint(square_size_range[0], square_size_range[1])
            square_position = (np.random.randint(0, map_size[0] - square_size), np.random.randint(0, map_size[1] - square_size))
        forbidden_areas.append((square_position, square_size))
    return forbidden_areas
# ==================== End of logistic of forbidden areas generation ====================

forbidden_areas = generate_forbidden_areas(map_size, antenna_position,
                                           NUM_FORBIDDEN_SQUARES, 
                                           SQUARE_SIZE_RANGE,
                                           RANDOM_SEED)


# ==================== LOS/NLOS region computation ====================
def _build_obstacle_grid(map_size, forbidden_areas):
    """
    根据禁区生成障碍物网格，True 表示该格点为障碍物
    :param map_size: 地图尺寸 (rows, cols)
    :param forbidden_areas: 禁区列表 [((x, y), size), ...]
    :return: 2D 布尔数组，形状为 map_size
    """
    rows, cols = map_size[0], map_size[1]
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


def _bresenham_line(x0, y0, x1, y1):
    """
    Bresenham 直线算法，返回从 (x0,y0) 到 (x1,y1) 经过的格点坐标列表（不含终点）
    :return: list of (x, y)
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


def compute_los_nlos_regions(map_size, forbidden_areas, antenna_position):
    """
    计算地图中相对天线位置的 LOS（视距）与 NLOS（非视距）区域并返回。

    - LOS：从该格点到天线连线未穿过任何禁区（障碍物）的格点集合。
    - NLOS：禁区内的格点，或到天线连线被障碍物遮挡的格点集合。

    :param map_size: 地图尺寸 (rows, cols)，如 (48, 24)
    :param forbidden_areas: 禁区列表 [((x, y), size), ...]
    :param antenna_position: 天线位置 (x, y)，可为元组或数组
    :return: (los_region, nlos_region)
        - los_region: set of (x, y) 为 LOS 格点
        - nlos_region: set of (x, y) 为 NLOS 格点
    """
    rows, cols = map_size[0], map_size[1]
    ap = (int(antenna_position[0]), int(antenna_position[1]))
    obstacle_grid = _build_obstacle_grid(map_size, forbidden_areas)

    los_region = set()
    nlos_region = set()

    for i in range(rows):
        for j in range(cols):
            if obstacle_grid[i, j]:
                nlos_region.add((i, j))
                continue
            # 从 (i, j) 到天线的射线
            line_cells = _bresenham_line(i, j, ap[0], ap[1])
            # 检查射线经过的中间点（不含起点和终点）是否经过障碍
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


# 预计算 LOS/NLOS 区域（使用当前 map 与天线配置）
los_region, nlos_region = compute_los_nlos_regions(
    map_size, forbidden_areas, antenna_position
)


# ==================== LOS/NLOS 离散化网格与查询 ====================
def build_los_nlos_grid(map_size, los_region, nlos_region):
    """
    将 LOS/NLOS 区域离散化为 2D 网格，便于 O(1) 查询。
    :param map_size: 地图尺寸 (rows, cols)
    :param los_region: LOS 格点集合 set of (x, y)
    :param nlos_region: NLOS 格点集合 set of (x, y)
    :return: 2D 数组，形状为 map_size，1 表示 LOS，0 表示 NLOS
    """
    rows, cols = map_size[0], map_size[1]
    grid = np.zeros((rows, cols), dtype=np.int32)  # 默认 NLOS
    for (i, j) in los_region:
        if 0 <= i < rows and 0 <= j < cols:
            grid[i, j] = 1
    return grid


# 离散化 LOS/NLOS 网格（1=LOS, 0=NLOS）
los_nlos_grid = build_los_nlos_grid(map_size, los_region, nlos_region)


def get_los_nlos(x: int, y: int, los_nlos_grid_=None, map_size_=None):
    """
    根据离散化后的坐标 (x, y) 返回该格点的 LOS/NLOS 分类。
    :param x: 离散化行坐标（网格索引）
    :param y: 离散化列坐标（网格索引）
    :param los_nlos_grid_: 可选，LOS/NLOS 网格，默认使用模块级 los_nlos_grid
    :param map_size_: 可选，地图尺寸，默认使用模块级 map_size
    :return: 'los' 或 'nlos'
    """
    grid = los_nlos_grid_ if los_nlos_grid_ is not None else los_nlos_grid
    size = map_size_ if map_size_ is not None else map_size
    rows, cols = size[0], size[1]
    if x < 0 or x >= rows or y < 0 or y >= cols:
        return "nlos"
    return "los" if grid[int(x), int(y)] == 1 else "nlos"


# specify the variables to be exported
__all__ = [
    "map_size", "forbidden_areas",
    "RANDOM_SEED", "SQUARE_SIZE_RANGE", "NUM_FORBIDDEN_SQUARES",
    "compute_los_nlos_regions", "los_region", "nlos_region",
    "build_los_nlos_grid", "los_nlos_grid", "get_los_nlos",
]
