"""
训练曲线绘图：8 张图。

7 张 1x2 布局（左：全局/总和，叠 max 虚线；右：每 agent）：
    {prefix}_return.png         # 整体 return
    {prefix}_step_reward.png    # 时间惩罚累计
    {prefix}_approach_reward.png# 接近目标奖励累计（含 closer/farther/same/goal/forbidden）
    {prefix}_path_reward.png    # step + approach 之和；理想最短路径下趋向 +reward_goal
    {prefix}_ber_reward.png     # 通信奖励累计
    {prefix}_ber.png            # 每 ep 的 mean(-log10 BER)
    {prefix}_ber_best.png       # 每 ep 的 max(-log10 BER) ≡ -log10(min BER)，best signal quality reached

1 张单图（无 per-agent 拆分，因为 wall-clock 是全局指标）：
    {prefix}_time.png           # 每 ep 的 wall-clock 耗时（秒）

注意：path_reward 是 step_return + approach_return 的逐元素和。approach 里仍混着 goal(+50)
和 forbidden(-5)，所以这条曲线**不是纯路径长度估计**：撞禁区会下拉、到达 goal 会上拉
reward_goal。要得到完全纯净的"step + closer"信号，需要 env.step 拆出第 4 路 reward。
"""
import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_metric(values, agent_values, title_total, title_agents,
                 ylabel, out_path, highlight_max: bool = True):
    """1x2 布局：左整体（含 max 虚线），右分 agent。统一黑色/英文风格。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    episodes = range(1, len(values) + 1)
    ax1.plot(episodes, values, linewidth=2, label="Total")
    if highlight_max and values:
        vmax = max(values)
        ax1.axhline(y=vmax, color='r', linestyle='--', label=f"Max: {vmax:.2f}")
        ax1.text(len(values) * 0.02, vmax, f"Max: {vmax:.2f}",
                 fontsize=10, color='r', fontweight='bold')
    ax1.set_xlabel(f"Episode (total: {len(values)})")
    ax1.set_ylabel(ylabel)
    ax1.set_title(title_total)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    for idx, ag_vals in enumerate(agent_values):
        ax2.plot(range(1, len(ag_vals) + 1), ag_vals, label=f"Agent {idx + 1}")
    ax2.set_xlabel(f"Episode (total: {len(values)})")
    ax2.set_ylabel(ylabel)
    ax2.set_title(title_agents)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _fname(fig_dir: str, prefix: str, base: str) -> str:
    """prefix 为空时直接用 base 名；否则拼 ``<prefix>_<base>``。"""
    fname = f"{prefix}_{base}" if prefix else base
    return os.path.join(fig_dir, fname)


def _plot_single(values, title, ylabel, out_path,
                 highlight_max: bool = False, highlight_min: bool = False) -> str:
    """单面板绘图：x = Episode，y = values。Time-Ep 这种"全局标量"用此布局。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    episodes = range(1, len(values) + 1)
    ax.plot(episodes, values, linewidth=2, label="Per-episode")
    if highlight_max and values:
        vmax = max(values)
        ax.axhline(y=vmax, color='r', linestyle='--', label=f"Max: {vmax:.2f}")
    if highlight_min and values:
        vmin = min(values)
        ax.axhline(y=vmin, color='g', linestyle='--', label=f"Min: {vmin:.2f}")
    ax.set_xlabel(f"Episode (total: {len(values)})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_training(history: dict, fig_dir: str, prefix: str = "", algo_label: str = "MADQN") -> list:
    """
    :param history: trainer.train() 返回的 dict
    :param fig_dir: 输出目录（通常是 ``<run.dir>/figs/``）
    :param prefix : 文件名前缀；空字符串时直接落盘为 ``return.png`` 等
    :param algo_label: 图标题前缀，显示算法名
    """
    _ensure_dir(fig_dir)
    paths = []

    paths.append(_plot_metric(
        history["return_list"], history["agent_return_lists"],
        title_total=f"{algo_label} Total Return vs Episode",
        title_agents=f"{algo_label} Per-Agent Return vs Episode",
        ylabel="Return",
        out_path=_fname(fig_dir, prefix, "return.png"),
    ))

    paths.append(_plot_metric(
        history["step_return_list"], history["agent_step_return_lists"],
        title_total=f"{algo_label} Step Reward (time penalty) vs Episode",
        title_agents=f"{algo_label} Per-Agent Step Reward vs Episode",
        ylabel="Step Reward",
        out_path=_fname(fig_dir, prefix, "step_reward.png"),
        highlight_max=False,
    ))

    paths.append(_plot_metric(
        history["approach_return_list"], history["agent_approach_return_lists"],
        title_total=f"{algo_label} Approach Reward vs Episode",
        title_agents=f"{algo_label} Per-Agent Approach Reward vs Episode",
        ylabel="Approach Reward",
        out_path=_fname(fig_dir, prefix, "approach_reward.png"),
    ))

    # step + approach 合成：trainer 输出的两路 reward 逐元素相加。
    # 理想最短路径下每步 step+closer = -1 + 1 = 0，到达 goal 一次性 +reward_goal，
    # 所以 ep 总和 ≈ +reward_goal（默认 50）。撞禁区 / goal 会让这条曲线偏离纯路径估计。
    step_total = history["step_return_list"]
    approach_total = history["approach_return_list"]
    path_total = [s + a for s, a in zip(step_total, approach_total)]
    agent_step = history["agent_step_return_lists"]
    agent_approach = history["agent_approach_return_lists"]
    agent_path = [
        [s + a for s, a in zip(agent_step[i], agent_approach[i])]
        for i in range(len(agent_step))
    ]
    paths.append(_plot_metric(
        path_total, agent_path,
        title_total=f"{algo_label} Step + Approach Reward vs Episode",
        title_agents=f"{algo_label} Per-Agent Step + Approach Reward vs Episode",
        ylabel="Step + Approach",
        out_path=_fname(fig_dir, prefix, "path_reward.png"),
    ))

    paths.append(_plot_metric(
        history["comm_return_list"], history["agent_comm_return_lists"],
        title_total=f"{algo_label} Communication (BER) Reward vs Episode",
        title_agents=f"{algo_label} Per-Agent Communication Reward vs Episode",
        ylabel="BER Reward",
        out_path=_fname(fig_dir, prefix, "ber_reward.png"),
    ))

    paths.append(_plot_metric(
        history["ber_list"], history["agent_ber_lists"],
        title_total=f"{algo_label} Mean(-log10 BER) vs Episode",
        title_agents=f"{algo_label} Per-Agent Mean(-log10 BER) vs Episode",
        ylabel="-log10(BER)",
        out_path=_fname(fig_dir, prefix, "ber.png"),
    ))

    # 每 ep 的最佳信号质量：max(-log10 BER) ≡ -log10(min BER)；与 mean 对照看"峰值"通信窗口
    ber_max_list = history.get("ber_max_list") or []
    agent_ber_max_lists = history.get("agent_ber_max_lists") or []
    if ber_max_list and agent_ber_max_lists:
        paths.append(_plot_metric(
            ber_max_list, agent_ber_max_lists,
            title_total=f"{algo_label} Best (-log10 BER) vs Episode",
            title_agents=f"{algo_label} Per-Agent Best (-log10 BER) vs Episode",
            ylabel="-log10(BER)",
            out_path=_fname(fig_dir, prefix, "ber_best.png"),
        ))

    # Time-Ep：单图。time_list 缺失时跳过，向后兼容旧 history dict。
    time_values = history.get("time_list") or []
    if time_values:
        paths.append(_plot_single(
            time_values,
            title=f"{algo_label} Wall-Clock Time per Episode",
            ylabel="Time (s)",
            out_path=_fname(fig_dir, prefix, "time.png"),
            highlight_max=True,
            highlight_min=True,
        ))

    return paths
