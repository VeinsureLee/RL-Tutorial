"""
NOMA SIC 解码、二进制功率控制、SINR/BER 计算。
对齐论文：功率作为动作空间的一部分，每簇两用户按强弱分配不同功率等级。
"""
import numpy as np
from scipy.special import erfc
from scipy.stats import norm


def get_power_levels(P_max_per_cluster, num_levels):
    """
    生成二进制功率控制的功率等级列表（论文 3.2 节）。

    Args:
        P_max_per_cluster: 每簇最大功率 (mW)
        num_levels: 可用功率数 p

    Returns:
        strong_powers: 较强用户的功率等级列表，长度 p
                       P_max/2^p, ..., P_max/2^{2p} (从大到小排列，但都较小)
        weak_powers:   较弱用户的功率等级列表，长度 p
                       P_max/2, ..., P_max/2^{p} (从大到小排列，都较大)
    """
    p = num_levels
    # 较强机器人（信道好）分配较少功率
    strong_powers = np.array([P_max_per_cluster / (2 ** i) for i in range(p, 2 * p)])
    # 较弱机器人（信道差）分配较多功率
    weak_powers = np.array([P_max_per_cluster / (2 ** i) for i in range(1, p + 1)])
    return strong_powers, weak_powers


def compute_sinr(H_precoded, powers_strong, powers_weak, noise_power):
    """
    计算 SIC 后的 SINR（论文式 2-12, 2-13）。

    Args:
        H_precoded: (M, 2) array, |h_{m,i} * w_m|^2，M 簇各 2 用户的等效信道增益
        powers_strong: (M,) array, 每簇强用户分配功率 (mW)
        powers_weak: (M,) array, 每簇弱用户分配功率 (mW)
        noise_power: 噪声功率 (mW)

    Returns:
        sinr_strong: (M,) 强用户 SINR (SIC 后无簇内干扰)
        sinr_weak: (M,) 弱用户 SINR (含簇内干扰)
    """
    g_strong = H_precoded[:, 0]  # |h_{m,1} w_m|^2
    g_weak = H_precoded[:, 1]    # |h_{m,2} w_m|^2

    # 强用户 SIC 后 (式 2-12)
    sinr_strong = (powers_strong * g_strong) / noise_power

    # 弱用户 (式 2-13)
    sinr_weak = (powers_weak * g_weak) / (powers_strong * g_weak + noise_power)

    return sinr_strong, sinr_weak


def compute_ber(sinr, N, D):
    """
    有限块长下的解码错误概率（论文式 2-14 ~ 2-16）。

    Args:
        sinr: SINR 值 (线性，非 dB)
        N: 信道块长度
        D: 数据包大小 (bits)

    Returns:
        ber: 误码率 epsilon
    """
    sinr = np.maximum(sinr, 1e-10)  # 避免 log(0)

    # 信道色散 V (式 2-16)
    V = 1.0 - (1.0 + sinr) ** (-2)
    V = np.maximum(V, 1e-10)

    # 编码率
    rate = D / N

    # Q 函数参数 (式 2-14)
    capacity = np.log2(1.0 + sinr)
    xi = np.log(2) * np.sqrt(N / V) * (capacity - rate)

    # Q 函数: Q(x) = 0.5 * erfc(x / sqrt(2))
    ber = 0.5 * erfc(xi / np.sqrt(2))

    # 裁剪到 [1e-20, 1.0]
    ber = np.clip(ber, 1e-20, 1.0)

    return ber


def ber_to_reward(ber, reward_min=-1.0, reward_max=1.0, ber_worst=0.5, ber_best=1e-10):
    """
    BER 转奖励（论文式 3-2 改进）: R_rate = -log10(epsilon)，归一化并截断到 [reward_min, reward_max]。

    原始 -log10(BER) 范围极大（BER=1e-20→20, BER=0.5→0.3），
    为平衡导航与通信奖励，先归一化到 [0,1] 再映射到 [reward_min, reward_max]。

    Args:
        ber: 误码率 array
        reward_min: 最小奖励（通信质量差时）
        reward_max: 最大奖励（通信质量好时）
        ber_worst: 最差 BER 基准（对应 reward_min）
        ber_best: 最好 BER 基准（对应 reward_max）

    Returns:
        reward: 截断到 [reward_min, reward_max] 的奖励
    """
    ber = np.clip(ber, 1e-20, 1.0)
    raw = -np.log10(ber)
    # 使用参数化的基准进行归一化
    raw_worst = -np.log10(ber_worst)
    raw_best = -np.log10(ber_best)
    normalized = (raw - raw_worst) / (raw_best - raw_worst)
    normalized = np.clip(normalized, 0.0, 1.0)
    # 映射到 [reward_min, reward_max]
    return reward_min + normalized * (reward_max - reward_min)
