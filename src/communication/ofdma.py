"""
OFDMA 通信模型：每个 agent 占用独立正交子载波，无簇间/簇内干扰。
对齐原论文式 (2-17)(2-18)。
"""
import numpy as np
from communication.precompute import PrecomputedRadioMap
from communication.SIC import compute_ber, ber_to_reward


def compute_ber_rewards_ofdma(
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
    OFDMA BER 奖励计算。

    Args:
        radio_map: PrecomputedRadioMap 实例
        positions: (K, 2) int array，agent 网格坐标
        power_actions: (K,) int array，功率等级索引
        P_sum: 总发射功率 (mW)
        num_power_levels: 可用功率等级数 p
        N: 信道块长度
        D: 数据包大小 (bits)
        noise_power: 总噪声功率 (mW)
        rng: numpy random generator

    Returns:
        dict with:
            ber:   (K,) 每个 agent 的 BER
            sinr:  (K,) 每个 agent 的 SNR（OFDMA 无干扰，即 SNR）
            reward:(K,) 每个 agent 的通信奖励
    """
    positions = np.array(positions, dtype=int)
    K = len(positions)

    # 每个 agent 分配的最大功率 = P_sum / K
    P_per_agent = P_sum / K

    # 可用功率等级（与 NOMA strong_powers 结构对称）
    p = num_power_levels
    power_levels = np.array([P_per_agent / (2 ** i) for i in range(1, p + 1)])  # 降序：最大 P/2，最小 P/2^p

    # 子载波噪声功率：总噪声 / K
    noise_per_sub = noise_power / K

    # 获取信道向量（含 Rayleigh 衰落）
    H = radio_map.get_channel_vectors(positions, rng=rng)  # (K, N_t)

    # 每个 agent 独立 SNR，无干扰
    # 等效信道增益 = |h_k|^2（直接用向量模平方，无需预编码）
    gains = np.sum(np.abs(H) ** 2, axis=1)  # (K,)

    # 按功率动作选取功率
    p_indices = np.clip(power_actions, 0, len(power_levels) - 1)
    powers = power_levels[p_indices]  # (K,)

    snr = powers * gains / noise_per_sub  # (K,)

    # 有限块长 BER：带宽缩小 K 倍，等效码率 = K*D/N
    effective_D = K * D
    ber_all = compute_ber(snr, N, effective_D)

    reward_all = ber_to_reward(
        ber_all,
        reward_min=ber_reward_min,
        reward_max=ber_reward_max,
        ber_worst=ber_worst,
        ber_best=ber_best,
    )

    return {
        "ber": ber_all,
        "sinr": snr,
        "reward": reward_all,
    }
