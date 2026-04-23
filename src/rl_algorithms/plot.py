"""
训练曲线绘图：5 张图，1x2 布局（左：全局/总和；右：每 agent）。
与历史格式保持一致：总和曲线叠加红色虚线与最大值标注。

输出：
    {prefix}_return.png         # 整体 return
    {prefix}_step_reward.png    # 时间惩罚累计
    {prefix}_approach_reward.png# 接近目标奖励累计
    {prefix}_ber_reward.png     # 通信奖励累计
    {prefix}_ber.png            # 每 ep 的 mean(-log10 BER)
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

    return paths
