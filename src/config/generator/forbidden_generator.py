
"""
功能：障碍物生成。生成不重叠、不覆盖天线、与天线保持安全距离、且互相分散的方形禁区。

设计要点（解决"障碍物挤在天线附近 / 互相挨着 / 居中遮挡"三种退化情况）：
    - antenna_keepout_margin：障碍 bbox 距离天线 (Chebyshev / L_inf) 的最小格距，
      避免障碍紧贴天线导致几乎所有方向的视线都被阻塞。
    - min_obstacle_spacing  ：任意两个障碍 bbox 之间的最小 Chebyshev 间距，
      逼迫障碍在地图上分散，而不是聚成一团。
    - 若约束过紧导致采样多次失败，会按阶段渐进放松（先放松 spacing，再放松 keepout），
      确保即使配置不合理也能落盘出 K 个障碍而不是死循环。
"""
from typing import Union, List, Tuple
import numpy as np


# ---------------------------------------------------------------- 几何工具

def _square_contains(square_position: Tuple[int, int], size: int, point: Tuple[int, int]) -> bool:
    """禁区方块是否包含某点（天线不能落在禁区内）。"""
    r, c = point[0], point[1]
    r0, c0 = square_position[0], square_position[1]
    return r0 <= r < r0 + size and c0 <= c < c0 + size


def _bbox_chebyshev_to_point(r0: int, c0: int, size: int, point: Tuple[int, int]) -> int:
    """
    bbox 任意单元格到 point 的最小 Chebyshev (L_inf) 距离。
    bbox 占据 [r0, r0+size-1] × [c0, c0+size-1]；point=(pr, pc)。
    若 point 落在 bbox 内返回 0。
    """
    pr, pc = int(point[0]), int(point[1])
    dr = max(r0 - pr, pr - (r0 + size - 1), 0)
    dc = max(c0 - pc, pc - (c0 + size - 1), 0)
    return max(dr, dc)


def _bbox_chebyshev_gap(a_r0: int, a_c0: int, a_sz: int,
                        b_r0: int, b_c0: int, b_sz: int) -> int:
    """
    两个 bbox 之间最小 Chebyshev 间距（重叠返回 0；相邻不重叠返回 1）。
    用于实现"任意两障碍至少隔开 spacing 格"的约束。
    """
    # 行向间距：若区间重叠则为 0，否则为最近端点距离
    gap_r = max(a_r0 - (b_r0 + b_sz - 1), b_r0 - (a_r0 + a_sz - 1), 0)
    gap_c = max(a_c0 - (b_c0 + b_sz - 1), b_c0 - (a_c0 + a_sz - 1), 0)
    # 两 bbox 的最近 Chebyshev 距离 = max(gap_r, gap_c)；
    # 若两个轴向都有间隔，最近格之间的 L_inf 仍是 max（对角接近）
    if gap_r == 0 and gap_c == 0:
        return 0  # 重叠或共享边/角
    return max(gap_r, gap_c)


# ---------------------------------------------------------------- 单次候选合法性

def _candidate_ok(r0: int, c0: int, size: int,
                  existing: List[Tuple[Tuple[int, int], int]],
                  antenna: Tuple[int, int],
                  rows: int, cols: int,
                  keepout_margin: int,
                  spacing: int) -> bool:
    """
    候选 bbox 是否同时满足：
        (1) 不越界；
        (2) 不覆盖天线（含 keepout_margin 内的安全圈）；
        (3) 与已有障碍 Chebyshev 距离 >= spacing。
    """
    if r0 < 0 or c0 < 0 or r0 + size > rows or c0 + size > cols:
        return False
    # 天线 keep-out：要求 bbox 到天线的距离 >= keepout_margin
    if _bbox_chebyshev_to_point(r0, c0, size, antenna) < max(1, keepout_margin):
        return False
    for (pos, sz) in existing:
        if _bbox_chebyshev_gap(r0, c0, size, pos[0], pos[1], sz) < spacing:
            return False
    return True


# ---------------------------------------------------------------- 主入口

def generate_forbidden_areas(
    map_size: Union[tuple, list, np.ndarray],
    antenna_position: Union[tuple, list, np.ndarray],
    num_forbidden_squares: int,
    square_size_range: Tuple[int, int],
    random_seed: int,
    antenna_keepout_margin: int = 0,
    min_obstacle_spacing: int = 0,
    max_attempts_per_square: int = 2000,
) -> List[Tuple[Tuple[int, int], int]]:
    """
    生成不重叠、与天线保持安全距离、且互相分散的方形禁区。

    :param antenna_keepout_margin: 障碍 bbox 距离天线的最小 Chebyshev 格数（>=1 表示
        障碍 bbox 不能与天线相邻；越大障碍越不会出现在地图中央）。0 / None 退化为
        旧行为（仅"不直接覆盖天线"）。
    :param min_obstacle_spacing : 任意两障碍 bbox 之间的最小 Chebyshev 间距（>=1 表示
        障碍 bbox 之间至少留 1 格空白，避免连成一片）。
    :param max_attempts_per_square: 单个障碍最大尝试次数；若约束过紧会按阶段放松。
    :return: [((row, col), size), ...]
    """
    rng = np.random.default_rng(int(random_seed))
    rows, cols = int(map_size[0]), int(map_size[1])
    ap = (int(antenna_position[0]), int(antenna_position[1]))
    sz_lo, sz_hi = int(square_size_range[0]), int(square_size_range[1])
    # 兼容旧用法：randint(lo, hi) 半开区间，hi <= lo 时直接当成定值
    if sz_hi <= sz_lo:
        sz_hi = sz_lo + 1

    # 上界保护：若 keepout 过大会让全图都被排除掉
    sz_max_for_layout = sz_hi - 1
    keepout_cap = max(0, min(rows, cols) // 2 - sz_max_for_layout - 1)
    keepout_eff = min(int(antenna_keepout_margin), keepout_cap) if antenna_keepout_margin else 0
    spacing_eff = max(0, int(min_obstacle_spacing))

    forbidden: List[Tuple[Tuple[int, int], int]] = []

    for k in range(int(num_forbidden_squares)):
        placed = False
        for attempt in range(max_attempts_per_square):
            # 每 500 次失败触发一次"放松"：先放松 spacing，再放松 keepout，最后退化为旧逻辑
            phase = attempt // 500
            if phase == 0:
                cur_keepout, cur_spacing = keepout_eff, spacing_eff
            elif phase == 1:
                cur_keepout, cur_spacing = keepout_eff, max(0, spacing_eff // 2)
            elif phase == 2:
                cur_keepout, cur_spacing = max(1, keepout_eff // 2), 0
            else:
                cur_keepout, cur_spacing = 0, 0

            sz = int(rng.integers(sz_lo, sz_hi))
            # 旧行为：r0 in [0, rows - sz)，bbox 占 [r0, r0+sz-1]，落在 [0, rows-1]
            r0 = int(rng.integers(0, max(1, rows - sz)))
            c0 = int(rng.integers(0, max(1, cols - sz)))

            # cur_keepout=0 时 _candidate_ok 内部仍要求 >= 1（不直接覆盖天线），保留旧不变量
            if not _candidate_ok(r0, c0, sz, forbidden, ap, rows, cols, cur_keepout, cur_spacing):
                # 旧逻辑（cur_keepout=0 且 cur_spacing=0）下唯一的硬约束：不覆盖天线 + 不重叠
                if cur_keepout == 0 and cur_spacing == 0:
                    overlap = any(
                        not (r0 + sz <= p[0] or p[0] + s <= r0
                             or c0 + sz <= p[1] or p[1] + s <= c0)
                        for (p, s) in forbidden
                    ) or _square_contains((r0, c0), sz, ap)
                    if overlap:
                        continue
                else:
                    continue

            forbidden.append(((r0, c0), sz))
            placed = True
            break
        if not placed:
            raise RuntimeError(
                f"could not place forbidden square #{k + 1}/{num_forbidden_squares} "
                f"after {max_attempts_per_square} attempts; "
                f"loosen num_forbidden_squares / square_size_range / "
                f"antenna_keepout_margin / min_obstacle_spacing in map.yml"
            )
    return forbidden


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
