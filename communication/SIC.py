import numpy as np
from config.arguments import parser
from utils import *


noise_power_dbm = parser.parse_args().power_AWGN
noise_power = dbm2watt(noise_power_dbm)


def compute_sinr(h1, h2, w, P1, P2):
    sigma = noise_power
    g1 = np.abs(h1 @ w)**2
    g2 = np.abs(h2 @ w)**2

    sinr_1 = P1 * g1 / sigma

    sinr_2 = P2 * g2 / (P1 * g2 + sigma)

    return sinr_1, sinr_2
