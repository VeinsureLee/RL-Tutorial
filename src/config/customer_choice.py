"""
交互式 Agent 起终点选择工具（matplotlib 点击界面）。

此模块须在 env/env.py 与 rl_algorithms/plot.py 被导入前调用，因为这两个模块
会将 matplotlib 后端锁定为 Agg（非交互），而本模块需要弹出交互式图形窗口。
main.py 通过延迟导入保证调用顺序正确。

用法（由 main.py 在加载 env 之前调用）::
    from config.customer_choice import choose_placements
    num_agents, start_states, target_states = choose_placements(env_cfg)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple

matplotlib.rcParams['axes.unicode_minus'] = False

# 每个 agent 对应一种颜色，最多支持 8 个
_AGENT_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
]


def _expand_forbidden(forbidden_areas, rows: int, cols: int) -> set:
    """将 [((r,c), size), ...] 列表展开为所有禁区格子坐标集合。"""
    occupied = set()
    for area in forbidden_areas:
        if not (isinstance(area, (list, tuple)) and len(area) == 2):
            continue
        pos, size = area[0], area[1]
        if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
            continue
        r0, c0 = int(pos[0]), int(pos[1])
        sz = int(size) if isinstance(size, (int, float, np.integer)) else int(size[0])
        for dr in range(sz):
            for dc in range(sz):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < rows and 0 <= c < cols:
                    occupied.add((r, c))
    return occupied


def _draw_map(ax,
              rows: int, cols: int,
              forbidden_set: set,
              antenna_pos: Tuple[int, int],
              start_states: List[Tuple[int, int]],
              target_states: List[Tuple[int, int]],
              num_agents: int) -> None:
    """在 ax 上重绘地图底图及已标注的起终点。"""
    ax.clear()

    # 底图：白色自由格 + 深灰禁区
    img = np.ones((rows, cols, 3), dtype=np.float32)
    for r, c in forbidden_set:
        img[r, c] = [0.25, 0.25, 0.25]
    ax.imshow(img, origin='upper', aspect='equal',
              extent=[0, cols, rows, 0], interpolation='nearest')

    # 天线位置（金色星标）
    ar, ac = antenna_pos
    ax.plot(ac + 0.5, ar + 0.5, '*', color='gold', markersize=14,
            markeredgecolor='#333333', markeredgewidth=0.7, zorder=5)

    # 已选起点：圆形 + S{i} 标签
    for i, (sr, sc) in enumerate(start_states):
        clr = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        ax.plot(sc + 0.5, sr + 0.5, 'o', color=clr, markersize=10,
                markeredgecolor='black', markeredgewidth=0.8, zorder=6)
        ax.text(sc + 0.5, sr + 0.5, f'S{i + 1}',
                ha='center', va='center', fontsize=6,
                fontweight='bold', color='white', zorder=7)

    # 已选终点：方形 + T{i} 标签
    for i, (tr, tc) in enumerate(target_states):
        clr = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        ax.plot(tc + 0.5, tr + 0.5, 's', color=clr, markersize=10,
                markeredgecolor='black', markeredgewidth=0.8, zorder=6)
        ax.text(tc + 0.5, tr + 0.5, f'T{i + 1}',
                ha='center', va='center', fontsize=6,
                fontweight='bold', color='white', zorder=7)

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)   # row 0 在顶部，row N 在底部
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(True, linewidth=0.2, alpha=0.25, color='gray')

    # 图例
    handles = [
        mpatches.Patch(facecolor=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                       edgecolor='black', linewidth=0.5,
                       label=f'Agent {i + 1}')
        for i in range(num_agents)
    ]
    handles += [
        mpatches.Patch(facecolor=[0.25, 0.25, 0.25], label='Forbidden area'),
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='gold', markersize=10,
                   markeredgecolor='#333', label='Antenna'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markersize=8, label='S = start (circle)'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='gray', markersize=8, label='T = target (square)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              framealpha=0.9, borderpad=0.5)


def choose_placements(
    env_cfg: dict,
) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    弹出 matplotlib 可视化窗口，交互式点击选择每个 agent 的起点与终点。

    操作流程
    --------
    1. 控制台输入 agent 数量（1~8）。
    2. 地图窗口弹出，地图中白色=可用格，深灰=禁区，金星=天线位置。
    3. 依次为每个 agent 点击起点（圆形 S）与终点（方形 T）。
    4. 可用工具栏缩放/平移后再点击；无效点击（禁区/重复/越界）会被忽略。
    5. 全部选完后关闭窗口即可继续。

    Parameters
    ----------
    env_cfg : dict
        由 get_env_config() 返回，需含 map_size / antenna_position / forbidden_areas。

    Returns
    -------
    num_agents : int
    start_states : list of (row, col) tuples
    target_states : list of (row, col) tuples
    """
    rows, cols = env_cfg["map_size"]
    antenna_pos = tuple(int(x) for x in env_cfg["antenna_position"])
    forbidden_set = _expand_forbidden(env_cfg["forbidden_areas"], rows, cols)

    # 控制台询问 agent 数量
    while True:
        raw = input("Enter number of agents [1-8]: ").strip()
        try:
            n = int(raw)
            if 1 <= n <= 8:
                num_agents = n
                break
            print("  Must be between 1 and 8.")
        except ValueError:
            print("  Please enter a valid integer.")

    start_states: List[Tuple[int, int]] = []
    target_states: List[Tuple[int, int]] = []

    # 根据地图宽高比决定窗口尺寸（最大 14 英寸）
    aspect = cols / max(rows, 1)
    fig_h = float(min(14.0, max(8.0, rows / 8.0)))
    fig_w = float(min(14.0, max(6.0, fig_h * aspect + 2.5)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.suptitle(
        "Multi-Robot DRL — Agent Placement\n"
        "Zoom/Pan via toolbar  |  Click arrow icon to return to select mode",
        fontsize=9, y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    try:
        fig.canvas.manager.set_window_title("Agent Placement")
    except Exception:
        pass

    print("\nInstructions:")
    print("  - White cells = free,  dark gray = forbidden,  gold star = antenna")
    print("  - Use toolbar Zoom/Pan, then click the arrow (pointer) icon to re-enable clicking")
    print("  - Invalid clicks (forbidden / antenna / duplicate / out-of-bounds) are ignored\n")

    def _pick(prompt: str) -> Tuple[int, int]:
        """更新标题，循环等待用户点击一个合法格子并返回 (row, col)。"""
        while True:
            _draw_map(ax, rows, cols, forbidden_set, antenna_pos,
                      start_states, target_states, num_agents)
            ax.set_title(prompt, fontsize=10, pad=6)
            fig.canvas.draw()
            fig.canvas.flush_events()

            pts = plt.ginput(1, timeout=0, show_clicks=True)
            if not pts:
                print("  No click registered — please click on the map.")
                continue

            xc, yc = pts[0]
            c, r = int(xc), int(yc)

            if not (0 <= r < rows and 0 <= c < cols):
                print(f"  ({r}, {c}) is out of bounds — click inside the white grid.")
                continue
            if (r, c) in forbidden_set:
                print(f"  ({r}, {c}) is a forbidden cell (dark gray) — choose a white cell.")
                continue
            if (r, c) == antenna_pos:
                print(f"  ({r}, {c}) is the antenna cell — choose another cell.")
                continue
            if (r, c) in set(start_states) | set(target_states):
                print(f"  ({r}, {c}) is already assigned — choose a different cell.")
                continue

            return (r, c)

    for i in range(num_agents):
        lbl = f"Agent {i + 1}"

        s = _pick(f"[{lbl}]  Click START position  (S{i + 1})")
        start_states.append(s)
        print(f"  {lbl} start  -> row={s[0]}, col={s[1]}")

        t = _pick(f"[{lbl}]  Click TARGET position  (T{i + 1})")
        target_states.append(t)
        print(f"  {lbl} target -> row={t[0]}, col={t[1]}")

    # 最终确认视图：展示全部选点，等用户关闭窗口
    _draw_map(ax, rows, cols, forbidden_set, antenna_pos,
              start_states, target_states, num_agents)
    ax.set_title("All agents placed — close this window to continue", fontsize=10, pad=6)
    fig.canvas.draw()
    print("\nAll placements confirmed. Close the map window to begin training/testing.")
    plt.show(block=True)
    plt.close(fig)

    return num_agents, start_states, target_states
