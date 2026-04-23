"""单元测试：src/communication/。"""
import numpy as np
import pytest


# ---------------------------------------------------------------- BD 零空间

def test_bd_zero_interference_between_clusters():
    """matrix_cal: 对 ∀ i≠m，H_i · w_m ≈ 0（簇间零干扰）。"""
    from communication.diagonalization_precoding import matrix_cal
    rng = np.random.default_rng(42)
    Nt = 16
    # 4 个簇，每簇 2 个用户
    clusters = []
    for _ in range(4):
        H = (rng.normal(size=(2, Nt)) + 1j * rng.normal(size=(2, Nt))) / np.sqrt(2)
        clusters.append(H)

    for m in range(4):
        w_m = matrix_cal(clusters, m)  # (Nt, 2)
        for i, H_i in enumerate(clusters):
            if i == m:
                continue
            residue = np.linalg.norm(H_i @ w_m)
            # BD 在数值上残差 < 1e-8；放宽到 1e-6 以容忍 SVD 数值
            assert residue < 1e-6, f"cluster {i} interferes with w_{m}: |HW|={residue}"


def test_bd_inside_cluster_gain_positive():
    """第 m 簇的 H_m · w_m 非零（不会把信号本身投到零空间）。"""
    from communication.diagonalization_precoding import matrix_cal
    rng = np.random.default_rng(7)
    Nt = 16
    clusters = [
        (rng.normal(size=(2, Nt)) + 1j * rng.normal(size=(2, Nt))) / np.sqrt(2)
        for _ in range(3)
    ]
    w0 = matrix_cal(clusters, 0)
    gain = np.linalg.norm(clusters[0] @ w0)
    assert gain > 1e-6


# ---------------------------------------------------------------- URLLC BER

def test_ber_monotonic_in_sinr():
    """SINR 升高 → BER 应单调降（URLLC 有限码长）。"""
    from communication.SIC import compute_ber
    N, D = 256, 16
    sinr = np.array([0.1, 1.0, 5.0, 20.0, 100.0])
    ber = compute_ber(sinr, N, D)
    for a, b in zip(ber[:-1], ber[1:]):
        assert b <= a + 1e-12, f"ber not monotonic: {ber}"


def test_ber_in_valid_range():
    """BER ∈ [1e-20, 1.0] 且不含 NaN/Inf。"""
    from communication.SIC import compute_ber
    sinr = np.array([1e-6, 1e-3, 0.1, 1, 10, 100, 1e4])
    ber = compute_ber(sinr, N=256, D=16)
    assert np.all(ber >= 1e-20 - 1e-30)
    assert np.all(ber <= 1.0 + 1e-12)
    assert np.all(np.isfinite(ber))


def test_ber_to_reward_maps_endpoints():
    """最差 BER → reward_min；最好 BER → reward_max。"""
    from communication.SIC import ber_to_reward
    worst = 0.5
    best = 1e-10
    r_worst = float(ber_to_reward(np.array([worst]), reward_min=-1, reward_max=1,
                                  ber_worst=worst, ber_best=best)[0])
    r_best = float(ber_to_reward(np.array([best]), reward_min=-1, reward_max=1,
                                 ber_worst=worst, ber_best=best)[0])
    assert np.isclose(r_worst, -1.0)
    assert np.isclose(r_best, 1.0)


# ---------------------------------------------------------------- NOMA 功率表

def test_power_levels_strong_smaller_than_weak():
    """信道好的用户功率更小（NOMA 功率域可分）。"""
    from communication.SIC import get_power_levels
    P_max = 100.0
    strong, weak = get_power_levels(P_max, num_levels=3)
    assert len(strong) == 3 and len(weak) == 3
    # 对应等级比较
    assert np.all(strong < weak)


# ---------------------------------------------------------------- 分簇

def test_cluster_pairs_strong_with_weak():
    """cluster_agents: 信道增益降序后第 m 名 vs 第 M+m 名。"""
    from communication.ber_reward import cluster_agents
    rng = np.random.default_rng(3)
    Nt = 32
    K = 6
    # 构造已知增益梯度：第 k 个 agent 信道乘以 (k+1)
    H = (rng.normal(size=(K, Nt)) + 1j * rng.normal(size=(K, Nt))) / np.sqrt(2)
    for k in range(K):
        H[k] *= (k + 1)
    clusters, sorted_idx = cluster_agents(H, K)
    assert len(clusters) == K // 2
    # 每簇的 strong 应来自排序前半，weak 来自后半
    for strong, weak in clusters:
        g_strong = np.sum(np.abs(H[strong]) ** 2)
        g_weak = np.sum(np.abs(H[weak]) ** 2)
        assert g_strong > g_weak
