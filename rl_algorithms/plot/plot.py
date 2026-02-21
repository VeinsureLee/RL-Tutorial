"""
绘图模块：训练曲线等，保存至 rl_algorithms/figs。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 图表保存目录（与 plot.py 同级的 figs 文件夹）
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def _ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_dqn(return_list, save_prefix="dqn"):
    """绘制 DQN 训练回报曲线并保存到 rl_algorithms/figs。"""
    _ensure_fig_dir()
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    max_return = max(return_list) if return_list else 0
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(return_list) + 1), return_list, label="Episode Return")
    plt.axhline(y=max_return, color='r', linestyle='--', label=f"Max Return: {max_return:.2f}")
    plt.xlabel(f'Episode (Total: {len(return_list)})')
    plt.ylabel('Return')
    plt.title('DQN Training Return vs Episode')
    plt.grid(True)
    plt.legend()
    if return_list:
        plt.text(len(return_list) * 0.02, max_return * 1.02, f'Max: {max_return:.2f}',
                 fontsize=10, color='r', fontweight='bold')
    out_path = os.path.join(FIG_DIR, f"{save_prefix}_return.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_madqn(return_list, agent_return_lists, ber_list, agent_ber_lists, save_prefix="madqn"):
    """绘制 MADQN 训练回报与误码率曲线并保存到 rl_algorithms/figs。"""
    _ensure_fig_dir()
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    paths = []

    # 回报图：总回报 + 各 Agent 回报
    max_return = max(return_list) if return_list else 0
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(return_list) + 1), return_list, label="总回报 (Total Return)", linewidth=2)
    plt.axhline(y=max_return, color='r', linestyle='--', label=f"最大回报: {max_return:.2f}")
    plt.xlabel(f'Episode (总计: {len(return_list)})')
    plt.ylabel('Return')
    plt.title('MADQN 训练总回报 vs Episode')
    plt.grid(True, alpha=0.3)
    plt.legend()
    if return_list:
        plt.text(len(return_list) * 0.02, max_return * 1.02, f'最大: {max_return:.2f}',
                 fontsize=10, color='r', fontweight='bold')
    plt.subplot(1, 2, 2)
    for agent_id, agent_returns in enumerate(agent_return_lists):
        plt.plot(range(1, len(agent_returns) + 1), agent_returns, label=f'Agent {agent_id + 1}')
    plt.xlabel(f'Episode (总计: {len(return_list)})')
    plt.ylabel('Return')
    plt.title('每个Agent的回报 vs Episode')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, f"{save_prefix}_return.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(p1)

    # 误码率图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(ber_list) + 1), ber_list, label="平均误码率 (Avg BER)", linewidth=2)
    plt.xlabel(f'Episode (总计: {len(ber_list)})')
    plt.ylabel('BER')
    plt.title('MADQN 训练每回合平均误码率 vs Episode')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 2, 2)
    for agent_id, agent_bers in enumerate(agent_ber_lists):
        plt.plot(range(1, len(agent_bers) + 1), agent_bers, label=f'Agent {agent_id + 1}')
    plt.xlabel(f'Episode (总计: {len(ber_list)})')
    plt.ylabel('BER')
    plt.title('每个Agent每回合误码率 vs Episode')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, f"{save_prefix}_ber.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(p2)

    return paths
