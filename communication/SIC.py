import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.param_arguments import parser
from utils import *
from scipy.special import erfc


noise_power_dbm = parser.parse_args().power_AWGN
noise_power = dbm2watt(noise_power_dbm)
channel_block_length = parser.parse_args().channel_block_length
packet_size = parser.parse_args().packet_size


def compute_sinr(h1, h2, w, P1, P2, verbose=False):
    """
    图二 式(2-12)(2-13)：NOMA 下强/弱用户 SINR。
    强用户（SIC 后）：SINR_{m,1} = P_{m,1}|h_{m,1}^H w_m|^2 / σ^2；
    弱用户（受强用户干扰）：SINR_{m,2} = P_{m,2}|h_{m,2}^H w_m|^2 / (P_{m,1}|h_{m,2}^H w_m|^2 + σ^2)。
    :param h1: 强用户信道向量 (Nt,) 或 (1, Nt)
    :param h2: 弱用户信道向量 (Nt,) 或 (1, Nt)
    :param w: 预编码向量或矩阵 (Nt,) 或 (Nt, 1)；若为 (Nt, 2) 则取第一列作为公共波束
    :param P1: 强用户功率 (W)
    :param P2: 弱用户功率 (W)
    :param verbose: 是否打印 SINR 计算过程
    :return: (sinr_1, sinr_2) 标量
    """
    sigma = max(float(noise_power), 1e-12)
    h1 = np.asarray(h1, dtype=np.complex128).ravel()
    h2 = np.asarray(h2, dtype=np.complex128).ravel()
    w = np.asarray(w, dtype=np.complex128)
    if w.ndim == 2:
        w = w[:, 0].ravel()
    else:
        w = w.ravel()
    g1 = np.abs(np.dot(h1, w)) ** 2
    g2 = np.abs(np.dot(h2, w)) ** 2
    g1 = max(float(g1), 1e-20)
    g2 = max(float(g2), 1e-20)
    sinr_1 = P1 * g1 / sigma
    sinr_2 = P2 * g2 / (P1 * g2 + sigma)
    if verbose:
        print("  [SINR 计算] σ² = {:.6e} W,  g1 = |h1^H w|² = {:.6e},  g2 = |h2^H w|² = {:.6e}".format(sigma, g1, g2))
        print("  [SINR 计算] P1 = {:.6e} W,  P2 = {:.6e} W".format(P1, P2))
        print("  [SINR 计算] SINR_1 = P1*g1/σ² = {:.6e} = {:.2f} dB".format(sinr_1, 10 * np.log10(max(sinr_1, 1e-20))))
        print("  [SINR 计算] SINR_2 = P2*g2/(P1*g2+σ²) = {:.6e} = {:.2f} dB".format(sinr_2, 10 * np.log10(max(sinr_2, 1e-20))))
    return float(sinr_1), float(sinr_2)

def Q(xi):
    """图二 式(2-15)：Q 函数 Q(ξ) = 0.5*erfc(ξ/sqrt(2))"""
    return 0.5 * erfc(xi / np.sqrt(2))

def V_func(sinr):
    """图二 式(2-16)：信道色散 V = 1 - (1 + SINR)^(-2)"""
    return 1 - (1 + sinr)**(-2)

def epsilon(sinr):
    """图二 式(2-14)：有限块长下的解码错误概率（误码率）ε = Q(ln2*sqrt(N/V)*log2(1+SINR) - D/N)"""
    sinr = max(float(sinr), 1e-10)  # 避免 V=0 导致除零
    N = channel_block_length
    D = packet_size
    V = V_func(sinr)
    V = max(V, 1e-10)  # 避免 sqrt(N/V) 爆炸
    return float(Q(np.log(2) * np.sqrt(N / V) * np.log2(1 + sinr) - D / N))


def allocate_power(total_power, g_strong, g_weak, rho_min=None):
    """
    NOMA 功率分配：弱用户分配更多功率，且满足 SIC 约束 (P2-P1)*g_strong >= rho_min。
    :param total_power: 总功率 (W)
    :param g_strong: 强用户信道增益 |h_strong @ w|^2
    :param g_weak: 弱用户信道增益 |h_weak @ w|^2
    :param rho_min: 最小功率差要求，若为 None 则从 config 读取
    :return: (P1, P2) 强用户、弱用户功率
    """
    if rho_min is None:
        rho_min = parser.parse_args().rho_min
    # 弱用户分配更多：P2 > P1，且 (P2 - P1) * g_strong >= rho_min
    # P1 + P2 = total_power => P2 = total_power - P1
    # (total_power - 2*P1) * g_strong >= rho_min => P1 <= (total_power*g_strong - rho_min) / (2*g_strong)
    g_strong = max(g_strong, 1e-12)
    p1_max = (total_power * g_strong - rho_min) / (2 * g_strong) if g_strong > 0 else total_power / 2
    p1_max = min(p1_max, total_power * 0.5)  # P1 不超过一半
    p1_max = max(0.0, min(p1_max, total_power))
    # 取 P1 为较小值，使弱用户获得更多功率
    P1 = min(total_power * 0.25, p1_max)
    P1 = max(1e-12, P1)
    P2 = total_power - P1
    return float(P1), float(P2)


def ber_to_reward(ber):
    """误码率取对数取相反数作为 reward，即 -log(ber)。ber 过小时裁剪避免 -inf。"""
    ber = np.clip(float(ber), 1e-10, 1.0)
    return -np.log(ber)


def get_noma_powers(P_max, p, P_min=None):
    """
    分簇后功率分配：大组 p 个用户功率为 P_max/2^p, ..., P_max/2^(2p-1)；
    小组 p 个用户功率为 P_max/2, ..., P_max/2^p。同一簇内大组用户为强用户、小组用户为弱用户。
    :param P_max: 最大功率 (W)
    :param p: 簇数（大组/小组各有 p 个用户）
    :param P_min: 最小功率 (W)，None 则从 config 读取
    :return: (large_powers, small_powers)，各为长度 p 的数组
    """
    if P_min is None:
        P_min = parser.parse_args().P_min
    P_max = max(float(P_max), P_min)
    large_powers = np.array([P_max / (2 ** (p + k)) for k in range(p)], dtype=np.float64)
    small_powers = np.array([P_max / (2 ** (k + 1)) for k in range(p)], dtype=np.float64)
    large_powers = np.clip(large_powers, P_min, P_max)
    small_powers = np.clip(small_powers, P_min, P_max)
    return large_powers, small_powers