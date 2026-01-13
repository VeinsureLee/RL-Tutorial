from config.arguments import parser
import numpy as np
from utils import *
from scipy.special import erfc


def gaussian_q_function(x):
    """
    计算高斯Q函数
    :param x: 输入值（标量/数组）
    :return: Q(x)的计算结果
    """
    return 0.5 * erfc(x / np.sqrt(2))

def snr_calculate(powers, chn_vct_values):
    noise_power_dbm = parser.parse_args().power_AWGN
    noise_power = dbm2watt(noise_power_dbm)
    K = parser.parse_args().number_of_robots
    return powers*(chn_vct_values**2)/(K*noise_power)

def channel_dispersion():
    return

if __name__ == "__main__":
    powers = np.array([1,2,3])
    chn_vct_values = np.array([1,2,3])
    print(powers*chn_vct_values)
    print(snr_calculate(powers, chn_vct_values))