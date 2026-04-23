"""
tests/ 的共享辅助：自定义 env、脚本化动作、轨迹图保存、性能计时。
conftest.py 已经处理了 sys.path 注入，这里直接 import src/ 下的模块即可。
"""
from __future__ import annotations

import datetime
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


# ---- 方向常量（对应 env.yml: action_directions）---------------------------
# 索引 0=右, 1=下, 2=左, 3=上, 4=停
DIR_RIGHT = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_UP = 3
DIR_STAY = 4


# ---- 项目路径 --------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent


# ---- 自定义 env -----------------------------------------------------------

def make_custom_env(
    *,
    start: Tuple[int, int] = (60, 30),
    target: Tuple[int, int] = (0, 0),
    forbidden_areas: Optional[list] = None,
):
    """
    用项目默认 env_cfg 构造一个 K=1 的 env，但覆盖 start / target / forbidden。

    注意：forbidden_areas 改动会触发 radio_map 重新预计算（~1s）。如果只是想测
    禁区回退行为但不关心 LOS 真实性，请用 ``override_forbidden_set`` 后面直接改
    ``env.forbidden_set``，那样不会触发重算。
    """
    from env.env import MultiRobotEnv
    from config.yml_config import get_env_config

    cfg = dict(get_env_config())
    cfg["start_states"] = [tuple(start)]
    cfg["target_states"] = [tuple(target)]
    if forbidden_areas is not None:
        cfg["forbidden_areas"] = list(forbidden_areas)
    return MultiRobotEnv(cfg)


def override_forbidden_set(env, cells: Iterable[Tuple[int, int]]) -> None:
    """只改 env.forbidden_set（step 用它判定回退），不动 radio_map 的禁区。

    适合"看看撞禁区能不能正确回退"这种测试。
    """
    env.forbidden_set = {(int(r), int(c)) for r, c in cells}


# ---- 脚本化动作 -----------------------------------------------------------

def run_scripted(env, dir_sequence: List[int]) -> dict:
    """
    重置 env 并逐步执行一段方向序列。

    :param env: MultiRobotEnv 实例（K=1）
    :param dir_sequence: 方向索引列表（使用 DIR_UP / DIR_DOWN 等常量）
    :return: dict {
        positions: 从起点开始每步后的 (row, col) 轨迹，len = 1 + len(dir_sequence)
        step_rewards, approach_rewards, comm_rewards, totals: 每步累加
        dones_when: 某一步若 done 记录该步索引，否则 None
    }
    """
    env.reset()
    positions = [tuple(env.positions[0])]
    step_rewards = []
    approach_rewards = []
    comm_rewards = []
    totals = []
    dones_when = None
    n_powers = max(env.n_powers, 1)

    for i, dir_idx in enumerate(dir_sequence):
        # 复合动作 = dir_idx * n_powers + 0（固定选功率 0）
        action = int(dir_idx) * n_powers
        _, rewards, dones, info = env.step([action])
        positions.append(tuple(env.positions[0]))
        step_rewards.append(float(info["step_rewards"][0]))
        approach_rewards.append(float(info["approach_rewards"][0]))
        comm_rewards.append(float(info["comm_rewards"][0]))
        totals.append(float(rewards[0]))
        if dones[0] and dones_when is None:
            dones_when = i + 1  # 1-based step number

    return {
        "positions": positions,
        "step_rewards": step_rewards,
        "approach_rewards": approach_rewards,
        "comm_rewards": comm_rewards,
        "totals": totals,
        "dones_when": dones_when,
    }


# ---- 轨迹可视化 -----------------------------------------------------------

def viz_out_dir() -> Path:
    """实验产物目录：experiments/unit_tests/<today>/viz/"""
    today = datetime.date.today().strftime("%Y%m%d")
    d = _ROOT / "experiments" / "unit_tests" / today / "viz"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_trajectory_fig(env, title: str, filename: str, subtitle: str = "") -> Path:
    """
    用 env.render_nav_frame 画轨迹并保存 PNG。

    env.trajectories[0] 里已经记录了从 reset 起的全部位置，直接渲染即可。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = viz_out_dir() / filename
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    env.render_nav_frame(ax)
    # 叠加标题（render_nav_frame 已经设了 "Navigation Map" 标题，这里覆盖）
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- 性能计时 -------------------------------------------------------------

def time_it(name: str, fn, *, n_warmup: int = 2, n_iter: int = 50) -> dict:
    """
    计时工具：n_warmup 次热身 + n_iter 次测量，返回 dict。
    测试函数里用 print 输出，配合 pytest -s 可见。

    :return: {mean_ms, median_ms, min_ms, max_ms, n}
    """
    # 热身
    for _ in range(n_warmup):
        fn()
    times = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1000.0  # ms
    return {
        "name": name,
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n": n_iter,
    }


def print_timing(result: dict) -> None:
    """统一的性能输出格式。"""
    print(
        f"\n  [PERF] {result['name']}"
        f"  mean={result['mean_ms']:.3f} ms"
        f"  median={result['median_ms']:.3f} ms"
        f"  min={result['min_ms']:.3f} ms"
        f"  max={result['max_ms']:.3f} ms"
        f"  (n={result['n']})"
    )
