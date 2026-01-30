
'''
Specify parameters of the communication channel
'''
from typing import Union
import numpy as np
import argparse


parser = argparse.ArgumentParser("Settings for communication channel")


## ==================== User settings ====================
# specify the frequency of the carrier in GHz
parser.add_argument("--carrier-frequency", type=float, default=3.5)

# specify the sigma of rayleigh distribution (Usually set 1.2 in door)
parser.add_argument("--sigma_rayleigh", type=float, default=1.2)

# specify the number of antenna
parser.add_argument("--number_of_antenna", type=int, default=128)

# specify the antenna position in map
parser.add_argument("--antenna_position", type=Union[list, tuple, np.ndarray], default=(24,12))

# specify the power of AWGN in dbm/Hz
parser.add_argument("--power_AWGN", type=float, default=-143.0)

# specify the channel block length
parser.add_argument("--channel_block_length", type=int, default=256)

# specify the packet size
parser.add_argument("--packet_size", type=int, default=16)

# specify the number of robots
parser.add_argument("--number_of_robots", type=int, default=4)

# 总发射功率（瓦特），用于 NOMA 功率分配
parser.add_argument("--total_power", type=float, default=1.0)
# SIC 所需最小功率差 rho_min（瓦特），满足 (P2-P1)*g1 >= rho_min
parser.add_argument("--rho_min", type=float, default=0.01)
# 最大发射功率（瓦特），可选 50, 100, 150, 200, 250
parser.add_argument("--P_max", type=float, default=100.0, choices=[50.0, 100.0, 150.0, 200.0, 250.0])
# 最小发射功率（瓦特）
parser.add_argument("--P_min", type=float, default=5.0)
## ==================== End of User settings ====================


args = parser.parse_args()
