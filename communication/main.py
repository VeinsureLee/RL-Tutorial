import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.channel import (
    channel_parameter,
    channel_vector,
    channel_group,
    number_of_antenna,
    path_loss,
    path_loss_batch,
    path_loss_from_states,
)
from communication.diagonalization_precoding import matrix_cal
from communication.SIC import compute_sinr, epsilon, allocate_power, ber_to_reward, get_noma_powers

from config.param_arguments import parser as param_parser
from config.map_config import map_size as map_config_map_size, get_los_nlos as map_get_los_nlos

try:
    from env.env import Env
except ImportError:
    Env = None


def _get_grid_size():
    """从环境参数获取网格物理尺寸（米），若无则使用默认 0.4"""
    try:
        from config.env_arguments import env_parser
        return float(env_parser.parse_args().grid_size)
    except Exception:
        return 0.4


def get_ber_reward(agent_states, grid_size, antenna_position, get_los_nlos=None, P_max=None, verbose_sinr=False):
    """
    根据当前 agent 网格位置计算信道、分簇、功率分配与误码率，返回每个 agent 的 BER 及 reward。
    分簇：按信道向量范数分为大组/小组，大组第 k 个与小组第 k 个为一簇。
    功率：大组 P_max/2^p..P_max/2^(2p-1)，小组 P_max/2..P_max/2^p。
    :param agent_states: 列表，每个元素为 (grid_x, grid_y) 网格坐标
    :param grid_size: 网格对应的物理尺寸（米）
    :param antenna_position: 基站天线位置 (x, y)，与地图同一坐标尺度
    :param get_los_nlos: 可选，函数 (x, y) -> 'los'|'nlos'，用于每个格点；None 则全用 LOS
    :param P_max: 最大功率 (W)，None 则从 config 读取
    :param verbose_sinr: 是否打印每个簇的 SINR 计算过程
    :return: (rewards_ber, ber_per_agent, sinr_per_agent)，分别为 -log(ber) 列表、误码率列表、SINR 列表（线性值）
    """
    if P_max is None:
        P_max = param_parser.parse_args().P_max
    num_agents = len(agent_states)
    if num_agents == 0:
        return [], [], []

    if get_los_nlos is not None:
        models = [get_los_nlos(int(s[0]), int(s[1])).upper() for s in agent_states]
    else:
        models = ["LOS"] * num_agents
    # 天线与位置统一为物理距离（米）：config 中 antenna_position 为网格坐标，需乘以 grid_size
    antenna_phys = np.asarray(antenna_position, dtype=np.float64).reshape(2) * grid_size
    positions = np.array([[float(s[0]) * grid_size, float(s[1]) * grid_size] for s in agent_states], dtype=np.float64)
    distances = np.linalg.norm(positions - antenna_phys, axis=1)
    distances = np.maximum(distances, 1e-3)
    chn_paras = channel_parameter(distances, models=models)
    thetas = np.pi / 2
    chn_vcts = channel_vector(chn_paras, thetas)
    groups = channel_group(chn_vcts)
    large_group = groups["large_group"]
    small_group = groups["small_group"]
    p = min(len(large_group["indices"]), len(small_group["indices"]))
    cluster_list = [
        np.vstack([large_group["vectors"][k], small_group["vectors"][k]])
        for k in range(p)
    ]
    large_powers, small_powers = get_noma_powers(P_max, p)

    ber_per_agent = np.ones(num_agents) * 0.5
    sinr_per_agent = np.full(num_agents, np.nan, dtype=np.float64)

    for m in range(p):
        H_m = cluster_list[m]
        w_m = matrix_cal(cluster_list, m)
        if w_m.size == 0 or w_m.shape[1] < 2:
            continue
        w = w_m[:, 0]
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            continue
        w = w / w_norm
        h_strong = large_group["vectors"][m]
        h_weak = small_group["vectors"][m]
        P1 = large_powers[m]
        P2 = small_powers[m]
        if verbose_sinr:
            print("\n--- 簇 {} (强用户 agent {}, 弱用户 agent {}) ---".format(
                m, large_group["indices"][m], small_group["indices"][m]))
        sinr_1, sinr_2 = compute_sinr(h_strong, h_weak, w, P1, P2, verbose=verbose_sinr)
        ber_per_agent[large_group["indices"][m]] = epsilon(sinr_1)
        ber_per_agent[small_group["indices"][m]] = epsilon(sinr_2)
        sinr_per_agent[large_group["indices"][m]] = sinr_1
        sinr_per_agent[small_group["indices"][m]] = sinr_2

    if num_agents % 2 != 0 and len(small_group["indices"]) > p:
        idx_extra = small_group["indices"][p]
        H_single = small_group["vectors"][p : p + 1]
        cluster_list_single = cluster_list + [H_single]
        w_single = matrix_cal(cluster_list_single, p)
        if w_single.size > 0 and np.linalg.norm(w_single) > 1e-12:
            w_single = w_single[:, 0] / np.linalg.norm(w_single[:, 0])
            g = np.abs(np.dot(H_single.ravel(), w_single.ravel())) ** 2
            from utils import dbm2watt
            sigma = dbm2watt(param_parser.parse_args().power_AWGN)
            sigma = max(sigma, 1e-12)
            sinr = (P_max / 2) * g / sigma
            if verbose_sinr:
                print("\n--- 单用户 (agent {}) ---".format(idx_extra))
                print("  [SINR 计算] σ² = {:.6e} W,  g = |h^H w|² = {:.6e},  P = P_max/2 = {:.6e} W".format(sigma, g, P_max / 2))
                print("  [SINR 计算] SINR = P*g/σ² = {:.6e} = {:.2f} dB".format(sinr, 10 * np.log10(max(sinr, 1e-20))))
            ber_per_agent[idx_extra] = epsilon(sinr)
            sinr_per_agent[idx_extra] = sinr

    rewards_ber = [ber_to_reward(ber_per_agent[i]) for i in range(num_agents)]
    ber_list = [float(ber_per_agent[i]) for i in range(num_agents)]
    sinr_list = [float(sinr_per_agent[i]) for i in range(num_agents)]
    return rewards_ber, ber_list, sinr_list


def test_channel():
    distances = np.array([1, 10, 100])
    models_los = ["LOS", "LOS", "LOS"]
    models = ["LOS", "NLOS", "LOS"]
    models_nlos = ["NLOS", "NLOS", "NLOS"]
    pl_values = path_loss_batch(distances, models)
    pl_values_los = path_loss_batch(distances, models_los)
    pl_values_nlos = path_loss_batch(distances, models_nlos)
    print("Path Loss Values(dB):", pl_values)
    print("Path Loss Values LOS(dB):", pl_values_los)
    print("Path Loss Values NLOS(dB):", pl_values_nlos)
    chn_paras = channel_parameter(distances, models=models)
    print(chn_paras)
    thetas = [0, np.pi / 2, np.pi / 4]
    chn_vcts = channel_vector(chn_paras, thetas)
    print(len(chn_vcts[0]),len(chn_vcts))
    print(chn_vcts)
    groups = channel_group(chn_vcts)

    # 打印结果
    print("=== Large group ===")
    print("Prime index:", groups['large_group']['indices'])
    print("Norm:", groups['large_group']['norms'])
    print("Vector length:", len(groups['large_group']['vectors']))

    print("\n=== Small group ===")
    print("Prime index:", groups['small_group']['indices'])
    print("Norm:", groups['small_group']['norms'])
    print("Vector length:", len(groups['small_group']['vectors']))


def test_diag():
    # ========== 参数 ==========
    num_users_per_cluster = 6  # 每簇用户数（原来太小）
    num_clusters = 3

    distances = np.concatenate([
        np.full(num_users_per_cluster, 10),
        np.full(num_users_per_cluster, 50),
        np.full(num_users_per_cluster, 100),
    ])

    models = (
            ["LOS"] * num_users_per_cluster +
            ["NLOS"] * num_users_per_cluster +
            ["LOS"] * num_users_per_cluster
    )

    thetas = np.random.uniform(0, np.pi, size=len(distances))
    chn_paras = channel_parameter(distances, models=models)
    chn_vcts = channel_vector(chn_paras, thetas)
    groups = channel_group(chn_vcts)

    lg, sg = groups['large_group'], groups['small_group']
    p = min(len(lg['indices']), len(sg['indices']))
    cluster_list = [np.vstack([lg['vectors'][k], sg['vectors'][k]]) for k in range(p)]
    Nt = number_of_antenna

    for m in range(len(cluster_list)):
        print(f"\n=== Cluster {m + 1} ===")
        w_m = matrix_cal(cluster_list, m)

        for k, H_k in enumerate(cluster_list):
            interference = np.linalg.norm(H_k @ w_m, ord='fro')
            print(f"|| H_{k} @ w_{m} ||_F = {interference:.3e}")


def test_sic():
    """
    完整 BD + NOMA + SIC 测试
    1️⃣ 按物理簇做 Block Diagonalization (簇间干扰消除)
    2️⃣ 簇内按信道增益自动划分强/弱用户
    3️⃣ 自动跳过空簇或 BD 失败簇
    4️⃣ 打印簇间干扰和簇内强/弱用户 SINR
    """
    # ========== 参数 ==========
    num_users_per_cluster = 6
    num_clusters = 3

    # 距离和信道模型
    distances = np.concatenate([
        np.full(num_users_per_cluster, 10),
        np.full(num_users_per_cluster, 50),
        np.full(num_users_per_cluster, 100),
    ])

    models = (
        ["LOS"] * num_users_per_cluster +
        ["NLOS"] * num_users_per_cluster +
        ["LOS"] * num_users_per_cluster
    )

    thetas = np.random.uniform(0, np.pi, size=len(distances))

    # ---------- 信道生成 ----------
    chn_paras = channel_parameter(distances, models=models)
    chn_vcts = channel_vector(chn_paras, thetas)
    groups = channel_group(chn_vcts)

    # 物理簇列表：大组第 k 个与小组第 k 个为一簇
    lg, sg = groups['large_group'], groups['small_group']
    p = min(len(lg['indices']), len(sg['indices']))
    cluster_list = [np.vstack([lg['vectors'][k], sg['vectors'][k]]) for k in range(p)]

    # ---------- BD + NOMA + SIC ----------
    for m, H_m in enumerate(cluster_list):
        if H_m.size == 0:
            print(f"\nCluster {m+1}: empty cluster, skipping")
            continue

        print(f"\n========== Cluster {m + 1} ==========")

        # 1️⃣ BD 计算
        w_m = matrix_cal(cluster_list, m)

        if w_m.shape[1] == 0:
            print(f"Cluster {m+1}: BD failed (no null space), skipping SINR")
            continue

        # 2️⃣ 簇间干扰打印
        for k, H_k in enumerate(cluster_list):
            interf = np.linalg.norm(H_k @ w_m, ord='fro')
            print(f"|| H_{k} @ w_{m} ||_F = {interf:.3e}")

        # 3️⃣ 簇内 NOMA 波束选择（第一根波束）
        w = w_m[:, 0]
        w = w / np.linalg.norm(w)

        # 4️⃣ 簇内强/弱用户划分
        eff_gains = np.abs(H_m @ w)**2
        sorted_idx = np.argsort(eff_gains)[::-1]  # 从大到小
        h_strong = H_m[sorted_idx[0], :]
        h_weak   = H_m[sorted_idx[1], :]

        # 5️⃣ SINR 计算
        sinr_1, sinr_2 = compute_sinr(
            h_strong,
            h_weak,
            w,
            P1=1.0,  # 强用户功率
            P2=100.0   # 弱用户功率
        )

        print(f"SINR strong user : {10 * np.log10(sinr_1):.2f} dB")
        print(f"SINR weak user   : {10 * np.log10(sinr_2):.2f} dB")


def test_ber_random_agents(seed=None):
    """
    测试：按 num_agent（number_of_robots）随机生成网格点，按分配规则分配功率并计算误码率。
    - 随机生成 num_agents 个不重复的网格坐标 (grid_x, grid_y)
    - 使用 get_ber_reward 内部的分簇与 NOMA 功率分配（大组/小组、get_noma_powers）
    - 输出每个 agent 的 BER 与平均 BER

    P_max 来源：config/param_arguments.py 的默认值，或命令行 --P_max。
    若修改了 param_arguments.py 的 P_max 但测试结果未变：需重新启动 Python 再运行
    （例如新开终端执行 python communication/main.py）；或在命令行传入：python communication/main.py --P_max 150
    """
    if seed is not None:
        np.random.seed(seed)
    # 每次调用时重新解析，确保能读到最新的 config（或命令行 --P_max）
    args = param_parser.parse_args()
    num_agents = args.number_of_robots
    antenna_position = np.asarray(args.antenna_position, dtype=np.float64).reshape(2)
    P_max = args.P_max
    grid_size = _get_grid_size()
    rows, cols = map_config_map_size[0], map_config_map_size[1]

    # 在网格内随机生成 num_agents 个不重复的格点
    all_cells = [(i, j) for i in range(rows) for j in range(cols)]
    if len(all_cells) < num_agents:
        raise ValueError(
            f"地图格点数 {len(all_cells)} 小于 agent 数 {num_agents}，请增大地图或减少 number_of_robots"
        )
    chosen = np.random.choice(len(all_cells), size=num_agents, replace=False)
    agent_states = [all_cells[k] for k in chosen]

    # 使用地图 LOS/NLOS 与分配规则计算 BER 与 SINR（预编码按图一两次 SVD，误码率按图二），并打印 SINR 计算过程
    rewards_ber, ber_list, sinr_list = get_ber_reward(
        agent_states,
        grid_size=grid_size,
        antenna_position=antenna_position,
        get_los_nlos=map_get_los_nlos,
        P_max=P_max,
        verbose_sinr=True,
    )

    print("========== 随机 Agent 误码率 / SINR 测试 ==========")
    print(f"Agent 数量: {num_agents}, 网格尺寸: {rows}x{cols}, grid_size={grid_size}m, P_max={P_max}W")
    print(f"天线位置 (网格坐标): ({antenna_position[0]:.0f}, {antenna_position[1]:.0f})")
    print("\n各 Agent 网格位置、SINR 与误码率:")
    for i in range(num_agents):
        sinr_val = sinr_list[i]
        sinr_db = 10 * np.log10(sinr_val) if sinr_val is not None and sinr_val > 0 and np.isfinite(sinr_val) else float('nan')
        sinr_str = f"SINR = {sinr_val:.4f} ({sinr_db:.2f} dB)" if np.isfinite(sinr_db) else "SINR = N/A"
        print(f"  Agent {i}: 网格 ({agent_states[i][0]}, {agent_states[i][1]})  {sinr_str}  BER = {ber_list[i]:.6e}  reward = {rewards_ber[i]:.4f}")
    avg_ber = float(np.nanmean(ber_list))
    valid_sinr = [s for s in sinr_list if s is not None and np.isfinite(s) and s > 0]
    avg_sinr = float(np.mean(valid_sinr)) if valid_sinr else float('nan')
    print(f"\n平均误码率: {avg_ber:.6e}")
    if np.isfinite(avg_sinr):
        print(f"平均 SINR: {avg_sinr:.4f} ({10 * np.log10(avg_sinr):.2f} dB)")
    else:
        print("平均 SINR: N/A")
    print("==================================================")
    return agent_states, ber_list, rewards_ber, sinr_list


if __name__ == '__main__':
    # test_channel()
    # test_sic()
    test_ber_random_agents(seed=42)