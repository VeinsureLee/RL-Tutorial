"""
通信子模块：信道、路径损耗、NOMA/SIC、对角化预编码与 BER 奖励计算。
"""
from communication.channel import (
    path_loss,
    path_loss_batch,
    path_loss_from_states,
    channel_parameter,
    channel_vector,
    channel_group,
    number_of_antenna,
)
from communication.ber_reward import get_ber_reward
from communication.SIC import (
    compute_sinr,
    epsilon,
    allocate_power,
    ber_to_reward,
    get_noma_powers,
)
from communication.utils import watt2dbm, dbm2watt, judge_los_nlos, distances_calculation
from communication.diagonalization_precoding import matrix_cal, build_W_matrix

__all__ = [
    "path_loss",
    "path_loss_batch",
    "path_loss_from_states",
    "channel_parameter",
    "channel_vector",
    "channel_group",
    "number_of_antenna",
    "get_ber_reward",
    "compute_sinr",
    "epsilon",
    "allocate_power",
    "ber_to_reward",
    "get_noma_powers",
    "watt2dbm",
    "dbm2watt",
    "judge_los_nlos",
    "distances_calculation",
    "matrix_cal",
    "build_W_matrix",
]
