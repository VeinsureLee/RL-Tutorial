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


def compute_sinr(h1, h2, w, P1, P2):
    sigma = noise_power
    g1 = np.abs(h1 @ w)**2
    g2 = np.abs(h2 @ w)**2

    sinr_1 = P1 * g1 / sigma

    sinr_2 = P2 * g2 / (P1 * g2 + sigma)

    return sinr_1, sinr_2

def Q(xi):
    # Q函数使用 erfc 近似: Q(x) = 0.5 * erfc(x / sqrt(2))
    return 0.5 * erfc(xi / np.sqrt(2))

def V_func(sinr):
    return 1 - (1 + sinr)**(-2)

def epsilon(sinr):
    """有限块长下的解码错误概率（近似误码率）"""
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