"""单元测试：OFDMA 通信模型 (src/communication/ofdma.py)。"""
import numpy as np
import pytest


def _make_radio_map():
    """构造最小化 PrecomputedRadioMap（10x10 地图，不触碰正式缓存文件）。"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from communication.precompute import PrecomputedRadioMap
    return PrecomputedRadioMap(
        map_size=(10, 10),
        grid_size=0.4,
        antenna_position=(5, 5),
        h_AP=2.0,
        h_robot=1.5,
        h_block=3.0,
        n_antenna=8,
        carrier_freq_ghz=3.5,
        forbidden_areas=[],
        sigma_rayleigh=1.2,
        cache_dir=None,
    )


def test_ofdma_output_shape():
    """返回 dict 中 ber/sinr/reward 均为 (K,) 形状。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    K = 4
    positions = np.array([[1, 1], [2, 3], [5, 6], [7, 8]])
    power_actions = np.array([0, 1, 0, 1])
    result = compute_ber_rewards_ofdma(
        rm, positions, power_actions,
        P_sum=100.0, num_power_levels=2,
        N=256, D=16, noise_power=1e-7,
        rng=np.random.default_rng(42),
    )
    assert result["ber"].shape == (K,)
    assert result["sinr"].shape == (K,)
    assert result["reward"].shape == (K,)


def test_ofdma_ber_in_valid_range():
    """每个 agent 的 BER ∈ [0, 1]，不含 NaN/Inf。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    positions = np.array([[1, 2], [3, 4], [6, 7], [8, 9]])
    power_actions = np.zeros(4, dtype=int)
    result = compute_ber_rewards_ofdma(
        rm, positions, power_actions,
        P_sum=100.0, num_power_levels=1,
        N=256, D=16, noise_power=1e-7,
        rng=np.random.default_rng(0),
    )
    assert np.all(np.isfinite(result["ber"]))
    assert np.all(result["ber"] >= 0.0)
    assert np.all(result["ber"] <= 1.0)


def test_ofdma_snr_positive():
    """SNR > 0（功率和信道增益均为正）。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    positions = np.array([[2, 2], [4, 4]])
    power_actions = np.zeros(2, dtype=int)
    result = compute_ber_rewards_ofdma(
        rm, positions, power_actions,
        P_sum=100.0, num_power_levels=1,
        N=256, D=16, noise_power=1e-7,
        rng=np.random.default_rng(1),
    )
    assert np.all(result["sinr"] > 0)


def test_ofdma_higher_power_lower_ber():
    """更高功率等级 → 更高 SNR → 更低 BER（统计平均，固定种子）。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    positions = np.array([[3, 3], [6, 6]])
    rng = np.random.default_rng(99)
    result_high = compute_ber_rewards_ofdma(
        rm, positions, np.zeros(2, dtype=int),  # 最高功率（index=0）
        P_sum=200.0, num_power_levels=3,
        N=256, D=16, noise_power=1e-7, rng=rng,
    )
    rng2 = np.random.default_rng(99)
    result_low = compute_ber_rewards_ofdma(
        rm, positions, np.full(2, 2, dtype=int),  # 最低功率（index=2）
        P_sum=200.0, num_power_levels=3,
        N=256, D=16, noise_power=1e-7, rng=rng2,
    )
    # 平均 BER：高功率 ≤ 低功率（数值上留少量容差）
    assert result_high["ber"].mean() <= result_low["ber"].mean() + 1e-6


def test_ofdma_k1_works():
    """K=1 时不应报错（单 agent 边界条件）。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    positions = np.array([[4, 4]])
    result = compute_ber_rewards_ofdma(
        rm, positions, np.array([0]),
        P_sum=100.0, num_power_levels=1,
        N=256, D=16, noise_power=1e-7,
        rng=np.random.default_rng(7),
    )
    assert result["ber"].shape == (1,)
