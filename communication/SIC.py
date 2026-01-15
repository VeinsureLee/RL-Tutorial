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
    N = channel_block_length
    D = packet_size
    V = V_func(sinr)
    return Q(np.log(2) * np.sqrt(N / V) * np.log2(1 + sinr) - D / N)