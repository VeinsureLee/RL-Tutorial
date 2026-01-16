"""
This file combines map settings and forbidden area generation
You can configure map parameters and random square forbidden areas
"""
from typing import Union
import numpy as np


# ==================== Map settings ====================
# specify the size of map
map_size: Union[list, tuple, np.ndarray] = (48, 24)

# specify the size of grid
grid_size: float = 0.4

# specify the positions of start states
start_states: Union[list, tuple, np.ndarray] = (0, 0)

# specify the positions of target states
target_state: Union[list, tuple, np.ndarray] = (4, 4)
# ==================== End of map settings ====================


# ==================== Forbidden areas basic settings ====================
# random seed
RANDOM_SEED = 42

# the number of forbidden areas(squares)
NUM_FORBIDDEN_SQUARES = 5

# range of square size
SQUARE_SIZE_RANGE = (3, 5)

# forbidden areas can't cover start/target states
PROTECTED_COORDS = [start_states, target_state]
# ==================== Forbidden areas settings ====================


# ==================== Logistic of forbidden areas generation ====================
def generate_random_square_forbidden_areas(map_size, num_squares, size_range, protected_coords, seed = 42):
    """
    生成随机正方形禁止区域，避免覆盖保护坐标

    Args:
        map_size: 地图尺寸 (rows, cols)
        num_squares: 正方形数量
        size_range: 正方形边长范围 (min_size, max_size)
        protected_coords: 禁止覆盖的坐标列表
        seed: 随机种子

    Returns:
        所有禁止区域的坐标列表
    """
    # 固定随机种子
    np.random.seed(seed)
    forbidden_states = []
    rows, cols = map_size
    min_size, max_size = size_range

    # 验证参数合法性
    assert min_size > 0 and max_size <= min(rows, cols), "正方形边长超出地图范围"
    assert num_squares > 0, "正方形数量必须大于0"

    for _ in range(num_squares):
        # 随机生成正方形边长
        square_size = np.random.randint(min_size, max_size + 1)
        # 随机生成正方形左上角坐标（确保正方形完全在地图内）
        max_row = rows - square_size
        max_col = cols - square_size
        start_row = np.random.randint(0, max_row + 1)
        start_col = np.random.randint(0, max_col + 1)

        # 生成正方形内的所有坐标
        square_coords = []
        for i in range(start_row, start_row + square_size):
            for j in range(start_col, start_col + square_size):
                square_coords.append((i, j))

        # 检查是否覆盖保护坐标，若覆盖则重新生成
        overlap = any(coord in square_coords for coord in protected_coords)
        if overlap:
            continue  # 跳过覆盖保护坐标的正方形

        # 将合法的正方形坐标加入禁止区域
        forbidden_states.extend(square_coords)

    # 去重（避免多个正方形重叠）
    forbidden_states = list(set(forbidden_states))
    return forbidden_states


# 生成禁止区域
forbidden_states = generate_random_square_forbidden_areas(
    map_size=map_size,
    num_squares=NUM_FORBIDDEN_SQUARES,
    size_range=SQUARE_SIZE_RANGE,
    protected_coords=PROTECTED_COORDS,
    seed=RANDOM_SEED
)

# ==================== 地图奖励配置 ====================
# 到达目标的奖励
reward_target: float = 10
# 进入禁止区域的惩罚
reward_forbidden: float = -5
# 每步行动的惩罚
reward_step: float = -1

# 对外暴露所有配置变量
__all__ = [
    "map_size", "grid_size", "start_states", "target_state",
    "forbidden_states", "reward_target", "reward_forbidden", "reward_step"
]