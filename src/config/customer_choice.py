"""
交互式 Agent 起终点选择工具，支持测试用例的保存与加载。

此模块须在 env/env.py 与 rl_algorithms/plot.py 被导入前调用，因为那两个模块
会将 matplotlib 后端锁定为 Agg（非交互）。main.py 通过延迟导入保证调用顺序。

测试用例存储于 config/test_cases/，每条用例包含：
    {name}_{timestamp}.json   —— 起终点坐标（可复用）
    {name}_{timestamp}.png    —— 地图预览图（方便浏览）

用法::
    from config.customer_choice import choose_placements
    num_agents, start_states, target_states = choose_placements(env_cfg)
"""
import os
import re
import json
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['axes.unicode_minus'] = False

_AGENT_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
]


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

def _get_cases_dir() -> str:
    """返回 config/test_cases/ 的绝对路径，不存在则自动创建。"""
    from utils.path_tool import get_abs_path
    d = get_abs_path(os.path.join("config", "test_cases"))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 测试用例 I/O
# ---------------------------------------------------------------------------

def _list_cases(cases_dir: str) -> List[dict]:
    """扫描目录，返回按文件名排序的用例元数据列表。"""
    cases = []
    if not os.path.isdir(cases_dir):
        return cases
    for fname in sorted(os.listdir(cases_dir)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(cases_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['_id'] = fname[:-5]
            data['_json_path'] = path
            data['_png_path'] = path[:-5] + '.png'
            cases.append(data)
        except Exception:
            pass
    return cases


def _save_case(cases_dir: str, env_cfg: dict,
               num_agents: int,
               start_states: List[Tuple[int, int]],
               target_states: List[Tuple[int, int]],
               name: str) -> None:
    """将当前放置结果保存为 JSON + PNG 预览。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r'[^\w\-]', '_', name)[:30]
    fname = f"{safe}_{timestamp}" if safe else f"case_{timestamp}"

    data = {
        "name": name or fname,
        "created_at": datetime.now().isoformat(),
        "map_size": list(env_cfg["map_size"]),
        "num_agents": num_agents,
        "start_states": [list(s) for s in start_states],
        "target_states": [list(t) for t in target_states],
    }
    json_path = os.path.join(cases_dir, fname + '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 生成预览 PNG
    rows, cols = env_cfg["map_size"]
    antenna_pos = tuple(int(x) for x in env_cfg["antenna_position"])
    forbidden_set = _expand_forbidden(env_cfg["forbidden_areas"], rows, cols)
    aspect = cols / max(rows, 1)
    fig, ax = plt.subplots(figsize=(max(4.0, 6.0 * aspect), 6.0))
    _draw_map(ax, rows, cols, forbidden_set, antenna_pos,
              start_states, target_states, num_agents,
              markersize=8, label_fontsize=6)
    ax.set_title(f"{data['name']}\n{data['created_at'][:10]}  |  {num_agents} agent(s)",
                 fontsize=9)
    fig.tight_layout()
    png_path = os.path.join(cases_dir, fname + '.png')
    fig.savefig(png_path, dpi=90, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {json_path}")
    print(f"  Preview: {png_path}")


# ---------------------------------------------------------------------------
# 地图绘制（共用）
# ---------------------------------------------------------------------------

def _expand_forbidden(forbidden_areas, rows: int, cols: int) -> set:
    """将 [((r,c), size), ...] 展开为所有禁区格子坐标集合。"""
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
              num_agents: int,
              markersize: int = 10,
              label_fontsize: int = 6,
              show_legend: bool = True) -> None:
    """在 ax 上绘制地图底图及已标注的起终点。"""
    ax.clear()

    img = np.ones((rows, cols, 3), dtype=np.float32)
    for r, c in forbidden_set:
        img[r, c] = [0.25, 0.25, 0.25]
    ax.imshow(img, origin='upper', aspect='equal',
              extent=[0, cols, rows, 0], interpolation='nearest')

    ar, ac = antenna_pos
    ax.plot(ac + 0.5, ar + 0.5, '*', color='gold',
            markersize=markersize * 1.4,
            markeredgecolor='#333333', markeredgewidth=0.7, zorder=5)

    for i, (sr, sc) in enumerate(start_states):
        clr = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        ax.plot(sc + 0.5, sr + 0.5, 'o', color=clr, markersize=markersize,
                markeredgecolor='black', markeredgewidth=0.8, zorder=6)
        ax.text(sc + 0.5, sr + 0.5, f'S{i + 1}',
                ha='center', va='center', fontsize=label_fontsize,
                fontweight='bold', color='white', zorder=7)

    for i, (tr, tc) in enumerate(target_states):
        clr = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        ax.plot(tc + 0.5, tr + 0.5, 's', color=clr, markersize=markersize,
                markeredgecolor='black', markeredgewidth=0.8, zorder=6)
        ax.text(tc + 0.5, tr + 0.5, f'T{i + 1}',
                ha='center', va='center', fontsize=label_fontsize,
                fontweight='bold', color='white', zorder=7)

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xlabel('Column', fontsize=max(6, label_fontsize + 1))
    ax.set_ylabel('Row', fontsize=max(6, label_fontsize + 1))
    ax.grid(True, linewidth=0.2, alpha=0.25, color='gray')

    if show_legend:
        handles = [
            mpatches.Patch(facecolor=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                           edgecolor='black', linewidth=0.5,
                           label=f'Agent {i + 1}')
            for i in range(num_agents)
        ]
        handles += [
            mpatches.Patch(facecolor=[0.25, 0.25, 0.25], label='Forbidden'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='gold', markersize=9,
                       markeredgecolor='#333', label='Antenna'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='gray', markersize=7, label='S=start'),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor='gray', markersize=7, label='T=target'),
        ]
        ax.legend(handles=handles, loc='upper right',
                  fontsize=max(5, label_fontsize),
                  framealpha=0.9, borderpad=0.4)


# ---------------------------------------------------------------------------
# 历史用例浏览与选择
# ---------------------------------------------------------------------------

def _show_case_gallery(cases: List[dict], env_cfg: dict) -> Optional[dict]:
    """
    显示所有已保存用例的缩略图网格。
    用户单击某个缩略图选中它（窗口自动关闭），或按 'n' 键选择新建。
    返回被选中的用例 dict；若选择新建则返回 None。
    """
    n = len(cases)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    rows_map, cols_map = env_cfg["map_size"]
    antenna_pos = tuple(int(x) for x in env_cfg["antenna_position"])
    forbidden_set = _expand_forbidden(env_cfg["forbidden_areas"], rows_map, cols_map)

    aspect = cols_map / max(rows_map, 1)
    cell_w = max(3.5, min(5.5, 14.0 / ncols))
    cell_h = cell_w / aspect + 1.2   # +1.2 for subplot title

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell_w * ncols, cell_h * nrows),
                             squeeze=False)
    fig.suptitle(
        "Saved Test Cases  —  click a case to load  |  press 'n' to create new",
        fontsize=10,
    )

    # 建立 axes id -> 用例索引的映射，供点击回调使用
    _ax_to_idx: dict = {}
    for i, case in enumerate(cases):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        _ax_to_idx[id(ax)] = i
        starts = [tuple(s) for s in case["start_states"]]
        targets = [tuple(t) for t in case["target_states"]]
        _draw_map(ax, rows_map, cols_map, forbidden_set, antenna_pos,
                  starts, targets, case["num_agents"],
                  markersize=6, label_fontsize=5, show_legend=False)
        date = case.get("created_at", "")[:10]
        na = case["num_agents"]
        ax.set_title(
            f"[{i + 1}] {case.get('name', case['_id'])}\n{date}  |  {na} agent(s)",
            fontsize=8,
        )

    # 隐藏多余的子图
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    try:
        fig.canvas.manager.set_window_title("Test Cases Gallery")
    except Exception:
        pass

    selected: List[Optional[int]] = [None]   # None=closed, -1=new, >=0=index

    def _on_click(event):
        if event.inaxes is None:
            return
        idx = _ax_to_idx.get(id(event.inaxes))
        if idx is not None:
            selected[0] = idx
            plt.close(fig)

    def _on_key(event):
        if event.key in ('n', 'N'):
            selected[0] = -1
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    print("  [Gallery] Click a test case to load it, or press 'n' to create a new one.")
    plt.show(block=True)

    if selected[0] is None or selected[0] == -1:
        return None
    return cases[selected[0]]


# ---------------------------------------------------------------------------
# 交互式新建放置
# ---------------------------------------------------------------------------

def _interactive_placement(
    env_cfg: dict,
) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """弹出地图窗口，让用户逐一点击每个 agent 的起点与终点。"""
    rows, cols = env_cfg["map_size"]
    antenna_pos = tuple(int(x) for x in env_cfg["antenna_position"])
    forbidden_set = _expand_forbidden(env_cfg["forbidden_areas"], rows, cols)

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

    aspect = cols / max(rows, 1)
    fig_h = float(min(14.0, max(8.0, rows / 8.0)))
    fig_w = float(min(14.0, max(6.0, fig_h * aspect + 2.5)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.suptitle(
        "Multi-Robot DRL — New Placement\n"
        "Zoom/Pan via toolbar  |  Click arrow icon to re-enable cell selection",
        fontsize=9, y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        fig.canvas.manager.set_window_title("Agent Placement")
    except Exception:
        pass

    print("\nInstructions:")
    print("  - White = free cell,  dark gray = forbidden,  gold star = antenna")
    print("  - Use toolbar Zoom/Pan; click the arrow icon to return to select mode")
    print("  - Invalid clicks (forbidden / antenna / duplicate / out-of-bounds) are ignored\n")

    def _pick(prompt: str) -> Tuple[int, int]:
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
                print(f"  ({r}, {c}) out of bounds — click inside the white grid.")
                continue
            if (r, c) in forbidden_set:
                print(f"  ({r}, {c}) is forbidden (dark gray) — choose a white cell.")
                continue
            if (r, c) == antenna_pos:
                print(f"  ({r}, {c}) is the antenna cell — choose another cell.")
                continue
            if (r, c) in set(start_states) | set(target_states):
                print(f"  ({r}, {c}) already assigned — choose a different cell.")
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

    # 最终确认图
    _draw_map(ax, rows, cols, forbidden_set, antenna_pos,
              start_states, target_states, num_agents)
    ax.set_title("Placement confirmed — close window to continue", fontsize=10, pad=6)
    fig.canvas.draw()
    print("\nAll agents placed. Close the map window to continue.")
    plt.show(block=True)
    plt.close(fig)

    return num_agents, start_states, target_states


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def choose_placements(
    env_cfg: dict,
) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    选择 agent 起终点放置方案，优先从历史测试用例中加载，也可新建并保存。

    流程
    ----
    1. 若存在已保存用例：询问是否加载。
       - 选 y：打开缩略图画廊，点击选中某用例（或按 n 键跳到新建）。
       - 选 n：直接进入新建流程。
    2. 新建：弹出地图窗口逐格点击，完成后询问是否保存为新用例。
       - 输入名称后回车 → 以该名称保存 JSON + PNG 预览。
       - 直接回车（空白）→ 跳过保存。

    Parameters
    ----------
    env_cfg : dict   由 get_env_config() 返回

    Returns
    -------
    num_agents : int
    start_states : list of (row, col) tuples
    target_states : list of (row, col) tuples
    """
    cases_dir = _get_cases_dir()
    cases = _list_cases(cases_dir)

    selected_case: Optional[dict] = None

    if cases:
        print(f"\nFound {len(cases)} saved test case(s).")
        while True:
            ans = input("Load a saved test case? [y/N]: ").strip().lower()
            if ans in ('y', 'yes'):
                selected_case = _show_case_gallery(cases, env_cfg)
                break
            if ans in ('n', 'no', ''):
                break
            print("  Enter 'y' or 'n'.")

    # 校验已选用例的地图尺寸
    if selected_case is not None:
        if tuple(selected_case["map_size"]) != tuple(env_cfg["map_size"]):
            print(f"  [warning] map_size mismatch "
                  f"({selected_case['map_size']} vs {list(env_cfg['map_size'])}) "
                  f"— creating new placement instead.")
            selected_case = None

    if selected_case is not None:
        num_agents = selected_case["num_agents"]
        start_states = [tuple(int(x) for x in s) for s in selected_case["start_states"]]
        target_states = [tuple(int(x) for x in t) for t in selected_case["target_states"]]
        print(f"Loaded: \"{selected_case.get('name', selected_case['_id'])}\"  "
              f"({num_agents} agents)")
        return num_agents, start_states, target_states

    # 新建放置
    num_agents, start_states, target_states = _interactive_placement(env_cfg)

    # 询问是否保存
    raw = input("\nSave as test case? [Enter a name to save, or press Enter to skip]: ").strip()
    if raw:
        _save_case(cases_dir, env_cfg, num_agents, start_states, target_states, name=raw)

    return num_agents, start_states, target_states
