import sys
import os
import numpy as np
from channel import channel_parameter, channel_vector, channel_group, number_of_antenna, path_loss
from diagonalization_precoding import matrix_cal
from SIC import compute_sinr, epsilon, allocate_power, ber_to_reward

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.param_arguments import parser as param_parser

try:
    from env.env import Env
except ImportError:
    Env = None


def get_ber_reward(agent_states, grid_size, antenna_position, total_power=None):
    """
    根据当前 agent 网格位置计算信道、功率分配与误码率，返回每个 agent 的 BER reward（-log(ber)）。
    :param agent_states: 列表，每个元素为 (grid_x, grid_y) 网格坐标
    :param grid_size: 网格对应的物理尺寸
    :param antenna_position: 基站天线位置 (x, y)，与地图同一坐标尺度
    :param total_power: 总发射功率 (W)，None 则从 config 读取
    :return: (rewards_ber, ber_per_agent)，分别为 -log(ber) 列表与误码率列表
    """
    if total_power is None:
        total_power = param_parser.parse_args().total_power

    num_agents = len(agent_states)
    if num_agents == 0:
        return [], []

    # 网格坐标 -> 物理坐标，并计算到基站距离
    positions = np.array([
        [float(s[0]) * grid_size, float(s[1]) * grid_size] for s in agent_states
    ], dtype=np.float64)
    antenna = np.asarray(antenna_position, dtype=np.float64).reshape(2,)
    distances = np.linalg.norm(positions - antenna, axis=1)
    distances = np.maximum(distances, 1e-3)  # 避免 log10(0)

    # 信道：全部 LOS；角度随机（与位置相关可用 np.arctan2）
    models = ["LOS"] * num_agents
    thetas = np.arctan2(
        positions[:, 1] - antenna[1],
        positions[:, 0] - antenna[0]
    )
    thetas = np.clip(thetas, -np.pi, np.pi)

    chn_paras = channel_parameter(distances, models=models)
    chn_vcts = channel_vector(chn_paras, thetas)
    groups = channel_group(chn_vcts)

    cluster_list = [
        groups['cluster1']['vectors'],
        groups['cluster2']['vectors'],
        groups['cluster3']['vectors']
    ]
    cluster_indices = [
        groups['cluster1']['indices'],
        groups['cluster2']['indices'],
        groups['cluster3']['indices']
    ]

    # 每个 agent 的误码率，先初始化为默认（单用户或未分配时使用）
    ber_per_agent = np.ones(num_agents) * 0.5  # 默认 0.5，reward 为 -log(0.5)

    for m, H_m in enumerate(cluster_list):
        if H_m.size == 0:
            continue

        w_m = matrix_cal(cluster_list, m)
        if w_m.shape[1] == 0:
            continue

        w = w_m[:, 0].copy()
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            continue
        w = w / w_norm

        eff_gains = np.abs(H_m @ w) ** 2
        sorted_local = np.argsort(eff_gains)[::-1]
        n_in_cluster = H_m.shape[0]
        agent_idx_in_cluster = cluster_indices[m]

        if n_in_cluster == 1:
            g1 = eff_gains[sorted_local[0]]
            try:
                from utils import dbm2watt
                sigma = dbm2watt(param_parser.parse_args().power_AWGN)
            except Exception:
                sigma = 1e-12
            sigma = max(sigma, 1e-12)
            sinr_1 = (total_power * g1) / sigma
            ber_1 = epsilon(sinr_1)
            ber_per_agent[agent_idx_in_cluster[0]] = ber_1
            continue

        h_strong = H_m[sorted_local[0], :]
        h_weak = H_m[sorted_local[1], :]
        g_strong = np.abs(h_strong @ w) ** 2
        g_weak = np.abs(h_weak @ w) ** 2
        P1, P2 = allocate_power(total_power, g_strong, g_weak)
        sinr_1, sinr_2 = compute_sinr(h_strong, h_weak, w, P1, P2)
        ber_strong = epsilon(sinr_1)
        ber_weak = epsilon(sinr_2)

        # 簇内按有效增益排序：强用户(前一半)用 ber_strong，弱用户(后一半)用 ber_weak
        half = (n_in_cluster + 1) // 2
        for rank, local_i in enumerate(sorted_local):
            global_agent_id = agent_idx_in_cluster[local_i]
            ber_per_agent[global_agent_id] = ber_strong if rank < half else ber_weak

    rewards_ber = [ber_to_reward(ber_per_agent[i]) for i in range(num_agents)]
    ber_list = [float(ber_per_agent[i]) for i in range(num_agents)]
    return rewards_ber, ber_list


def test_channel():
    distances = np.array([1, 10, 100])
    models_los = ["LOS", "LOS", "LOS"]
    models = ["LOS", "NLOS", "LOS"]
    models_nlos = ["NLOS", "NLOS", "NLOS"]
    pl_values = path_loss(distances, models=models)
    pl_values_los = path_loss(distances, models=models_los)
    pl_values_nlos = path_loss(distances, models=models_nlos)
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
    print("=== Cluster 1 ===")
    print("Prime index:", groups['cluster1']['indices'])
    print("Norm:", groups['cluster1']['norms'])
    print("Vector length:", len(groups['cluster1']['vectors']))

    print("\n=== Cluster 2 ===")
    print("Prime index:", groups['cluster2']['indices'])
    print("Norm:", groups['cluster2']['norms'])
    print("Vector length:", len(groups['cluster1']['vectors']))

    print("\n=== Cluster 3 ===")
    if groups['cluster3']:
        print("Prime index:", groups['cluster3']['indices'])
        print("Norm:", groups['cluster3']['norms'])
        print("Vector length:", len(groups['cluster1']['vectors']))


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

    cluster_list = [
        groups['cluster1']['vectors'],
        groups['cluster2']['vectors'],
        groups['cluster3']['vectors']
    ]
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

    # 物理簇列表
    cluster_list = [
        groups['cluster1']['vectors'],
        groups['cluster2']['vectors'],
        groups['cluster3']['vectors']
    ]

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


if __name__ == '__main__':
    # test_channel()
    test_sic()