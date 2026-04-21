"""
BER 奖励计算入口：整合预计算表、信道向量、分簇、预编码、SINR/BER。
env.step() 调用此模块的 compute_ber_rewards()。
"""
import numpy as np
from communication.precompute import PrecomputedRadioMap
from communication.SIC import compute_sinr, compute_ber, ber_to_reward, get_power_levels
from communication.diagonalization_precoding import matrix_cal


def cluster_agents(H, K):
    """
    NOMA 分簇：按信道增益 |h_k|^2 降序排列，第 m 名与第 K/2+m 名配对（论文 2.3.1）。

    Args:
        H: (K, N_t) complex，信道向量矩阵
        K: agent 数量

    Returns:
        clusters: list of (strong_idx, weak_idx)，每簇包含强弱用户的原始索引
        sorted_indices: 按信道增益降序排列的索引
    """
    gains = np.sum(np.abs(H) ** 2, axis=1)  # (K,) 信道增益
    sorted_indices = np.argsort(-gains)  # 降序

    M = K // 2
    clusters = []
    for m in range(M):
        strong_idx = sorted_indices[m]       # 信道好的
        weak_idx = sorted_indices[M + m]     # 信道差的
        clusters.append((strong_idx, weak_idx))

    return clusters, sorted_indices


def compute_ber_rewards(
    radio_map: PrecomputedRadioMap,
    positions,
    power_actions,
    P_sum,
    num_power_levels,
    N,
    D,
    noise_power,
    rng=None,
    ber_reward_min=-1.0,
    ber_reward_max=1.0,
    ber_worst=0.5,
    ber_best=1e-10,
):
    """
    完整 BER 奖励计算流程。

    Args:
        radio_map: PrecomputedRadioMap 实例
        positions: (K, 2) int array，agent 网格坐标
        power_actions: (K,) int array，每个 agent 选择的功率等级索引
        P_sum: 总发射功率 (mW)
        num_power_levels: 可用功率数 p
        N: 信道块长度
        D: 数据包大小
        noise_power: 噪声功率 (mW)
        rng: numpy random generator

    Returns:
        dict with:
            ber: (K,) 每个 agent 的 BER
            sinr: (K,) 每个 agent 的 SINR
            reward: (K,) 每个 agent 的通信奖励 R_rate
    """
    positions = np.array(positions, dtype=int)
    K = len(positions)

    # 1. 获取信道向量（查预计算表 + Rayleigh 衰落）
    H = radio_map.get_channel_vectors(positions, rng=rng)

    # 2. 处理 K=1 的特殊情况
    if K == 1:
        gain = np.sum(np.abs(H[0]) ** 2)
        P_max_cluster = P_sum
        strong_powers, _ = get_power_levels(P_max_cluster, num_power_levels)
        p_idx = min(power_actions[0], len(strong_powers) - 1)
        power = strong_powers[p_idx]
        sinr_val = (power * gain) / noise_power
        ber_val = compute_ber(np.array([sinr_val]), N, D)
        reward_val = ber_to_reward(
            ber_val,
            reward_min=ber_reward_min,
            reward_max=ber_reward_max,
            ber_worst=ber_worst,
            ber_best=ber_best,
        )
        return {
            "ber": ber_val,
            "sinr": np.array([sinr_val]),
            "reward": reward_val,
        }

    # 3. 分簇
    clusters, sorted_indices = cluster_agents(H, K)
    M = len(clusters)
    P_max_per_cluster = P_sum / M

    # 功率等级表
    strong_powers_table, weak_powers_table = get_power_levels(P_max_per_cluster, num_power_levels)

    # 4. BD 预编码
    # 构建每簇的信道矩阵
    H_clusters = []
    for s_idx, w_idx in clusters:
        H_clusters.append(np.vstack([H[s_idx:s_idx+1], H[w_idx:w_idx+1]]))  # (2, N_t)

    W = [matrix_cal(H_clusters, m) for m in range(M)]  # list of (N_t, N_m) precoding matrices

    # 5. 计算等效信道增益和 SINR
    ber_all = np.zeros(K)
    sinr_all = np.zeros(K)

    for m, (s_idx, w_idx) in enumerate(clusters):
        w_m = W[m]  # (N_t, 2) — col 0 for strong user, col 1 for weak user

        # NOMA: 两个用户共享同一个预编码向量（取第一列归一化）
        w_vec = w_m[:, 0:1]  # (N_t, 1)
        w_vec = w_vec / (np.linalg.norm(w_vec) + 1e-12)

        # 等效信道增益 |h * w|^2
        g_strong = float(np.abs(H[s_idx] @ w_vec) ** 2)
        g_weak = float(np.abs(H[w_idx] @ w_vec) ** 2)

        # 根据 agent 的功率动作选择功率
        p_s_idx = min(power_actions[s_idx], len(strong_powers_table) - 1)
        p_w_idx = min(power_actions[w_idx], len(weak_powers_table) - 1)
        p_strong = strong_powers_table[p_s_idx]
        p_weak = weak_powers_table[p_w_idx]

        # SINR
        sinr_s = float((p_strong * g_strong) / noise_power)
        sinr_w = float((p_weak * g_weak) / (p_strong * g_weak + noise_power))

        sinr_all[s_idx] = sinr_s
        sinr_all[w_idx] = sinr_w

        # BER
        ber_all[s_idx] = float(compute_ber(np.array([sinr_s]), N, D)[0])
        ber_all[w_idx] = float(compute_ber(np.array([sinr_w]), N, D)[0])

    # 处理 K 为奇数时落单的 agent
    if K % 2 == 1:
        last_idx = sorted_indices[-1]
        gain = np.sum(np.abs(H[last_idx]) ** 2)
        p_idx = min(power_actions[last_idx], len(strong_powers_table) - 1)
        power = strong_powers_table[p_idx]
        sinr_val = (power * gain) / noise_power
        sinr_all[last_idx] = sinr_val
        ber_all[last_idx] = float(compute_ber(np.array([sinr_val]), N, D)[0])

    reward_all = ber_to_reward(
        ber_all,
        reward_min=ber_reward_min,
        reward_max=ber_reward_max,
        ber_worst=ber_worst,
        ber_best=ber_best,
    )

    return {
        "ber": ber_all,
        "sinr": sinr_all,
        "reward": reward_all,
    }
