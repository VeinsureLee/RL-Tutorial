import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.param_arguments import parser
from config.env_arguments import env_parser
import math


sigma = parser.parse_args().sigma_rayleigh
freq = parser.parse_args().carrier_frequency
number_of_antenna = parser.parse_args().number_of_antenna
antenna_position = parser.parse_args().antenna_position
grid_size = float(env_parser.parse_args().grid_size)


def _distance_from_state(state, antenna_pos, grid_size_m):
    """
    由离散格点坐标与天线位置、网格边长计算物理距离（米）。
    :param state: 离散坐标 (x, y)
    :param antenna_pos: 天线离散坐标 (x, y)
    :param grid_size_m: 网格边长（米）
    :return: 距离（米），至少为 grid_size_m 以避免 log10(0)
    """
    x, y = state[0], state[1]
    ax, ay = antenna_pos[0], antenna_pos[1]
    d_grid = math.sqrt((x - ax) ** 2 + (y - ay) ** 2)
    d_m = grid_size_m * max(d_grid, 1e-6)
    return d_m


def path_loss(state, region_type, grid_size_m=None, antenna_pos=None):
    """
    单状态路径损耗：输入离散坐标与区域类别，输出 path loss (dB)。
    物理距离由离散坐标与 grid_size 计算得到。
    :param state: 离散化坐标 (x, y)，单个 state
    :param region_type: 区域类别，与 env.get_los_nlos 一致，'los' 或 'nlos'（大小写不敏感）
    :param grid_size_m: 网格边长（米），默认使用 env_arguments 中的 grid_size（如 0.4）
    :param antenna_pos: 基站/天线离散坐标 (x, y)，默认使用 param_arguments 中的 antenna_position
    :return: path loss (dB)，标量
    """
    grid_size_m = grid_size_m if grid_size_m is not None else grid_size
    antenna_pos = antenna_pos if antenna_pos is not None else antenna_position
    if hasattr(antenna_pos, '__len__') and len(antenna_pos) >= 2:
        antenna_pos = (float(antenna_pos[0]), float(antenna_pos[1]))
    else:
        antenna_pos = (float(antenna_position[0]), float(antenna_position[1]))

    d = _distance_from_state(state, antenna_pos, grid_size_m)
    model = str(region_type).strip().upper()
    if model == "LOS":
        pl = 31.87 + 21.50 * np.log10(d) + 19.0 * np.log10(freq)
    elif model == "NLOS":
        pl_los = 31.87 + 21.50 * np.log10(d) + 19.0 * np.log10(freq)
        pl_sl = 33 + 25.50 * np.log10(d) + 20 * np.log10(freq)
        pl = max(pl_los, pl_sl)
    else:
        raise ValueError("Unsupported path loss model: {}".format(region_type))
    return float(pl)


def path_loss_batch(distances, models):
    """
    批量路径损耗：对给定的距离列表和模型列表，计算所有 path loss。
    :param distances: 机器人到 BS 的距离列表（米），或一维数组
    :param models: 路径损耗模型类型列表，"LOS" 或 "NLOS"，长度与 distances 一致
    :return: path loss 列表，与输入一一对应
    """
    distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
    models = np.atleast_1d(models)
    n = len(distances)
    if len(models) != n:
        raise ValueError("distances 与 models 长度须一致")
    pl_list = []
    for i in range(n):
        d = max(float(distances[i]), 1e-6)
        m = str(models[i]).strip().upper()
        if m == "LOS":
            pl_list.append(31.87 + 21.50 * np.log10(d) + 19.0 * np.log10(freq))
        elif m == "NLOS":
            pl_los = 31.87 + 21.50 * np.log10(d) + 19.0 * np.log10(freq)
            pl_sl = 33 + 25.50 * np.log10(d) + 20 * np.log10(freq)
            pl_list.append(max(pl_los, pl_sl))
        else:
            raise ValueError("Unsupported path loss model: {}".format(m))
    return pl_list


def path_loss_from_states(states, models, grid_size_m=None, antenna_pos=None):
    """
    根据所有 agent 的离散状态计算 path loss（内部计算距离并统一计算损耗）。
    :param states: 离散坐标列表 [(x1,y1), (x2,y2), ...]
    :param models: 每个格点的路径损耗模型列表，"LOS" 或 "NLOS"
    :param grid_size_m: 网格边长（米），默认使用 config 中的 grid_size
    :param antenna_pos: 基站离散坐标 (x, y)，默认使用 config 中的 antenna_position
    :return: path loss 列表，与 states 一一对应
    """
    grid_size_m = grid_size_m if grid_size_m is not None else grid_size
    antenna_pos = antenna_pos if antenna_pos is not None else antenna_position
    if hasattr(antenna_pos, '__len__') and len(antenna_pos) >= 2:
        antenna_pos = (float(antenna_pos[0]), float(antenna_pos[1]))
    distances = [_distance_from_state(s, antenna_pos, grid_size_m) for s in states]
    return path_loss_batch(distances, models)


def channel_parameter(distances, models):
    samples = len(distances)
    fading_factors = np.random.rayleigh(size=samples, scale=sigma)
    fading_factors = np.clip(fading_factors, a_min=1e-10, a_max=None)
    path_loss_values = path_loss_batch(distances, models)
    chn_paras = np.array(path_loss_values) - 10 * np.log10(fading_factors)
    return chn_paras

def channel_vector(chn_paras, thetas=None):
    """
    由信道参数和角度计算信道向量。theta 默认 90 度（np.pi/2）。
    :param chn_paras: 一维数组 (n,)，每个元素为信道参数（dB）
    :param thetas: 标量或一维数组 (n,)，到达角（弧度）。标量时对所有用户使用同一角度
    :return: 信道矩阵 (n, Nt)
    """
    chn_paras = np.atleast_1d(np.asarray(chn_paras, dtype=np.float64))
    n_users = chn_paras.size
    if thetas is None:
        thetas = np.pi / 2  # 默认 90 度
    thetas = np.atleast_1d(np.asarray(thetas, dtype=np.float64))
    if thetas.size == 1:
        thetas = np.full(n_users, float(thetas.flat[0]))
    elif thetas.size != n_users:
        raise ValueError("thetas 长度须与 chn_paras 一致或为标量")
    beta = 10 ** (-chn_paras / 20)
    n = np.arange(number_of_antenna, dtype=np.float64)[np.newaxis, :]
    phase_base = -1j * np.pi * np.sin(thetas)[:, np.newaxis]
    alpha_transmit = np.exp(n * phase_base) / np.sqrt(number_of_antenna)
    chn_vcts = beta[:, np.newaxis] * alpha_transmit
    return chn_vcts

def channel_group(chn_vcts):
    """
    按信道向量范数分为两组：大的一组（范数大）、小的一组（范数小）。
    大组中的第 k 个与小组中的第 k 个为一簇（配对），即 large_group['indices'][k] 与
    small_group['indices'][k] 属于同一簇。
    :param chn_vcts: 信道矩阵 (n, Nt)
    :return: dict with 'large_group', 'small_group'，每组含 'vectors', 'norms', 'indices'
    """
    chn_vcts = np.asarray(chn_vcts, dtype=np.complex128)
    num_vectors = chn_vcts.shape[0]
    norms = np.linalg.norm(chn_vcts, axis=1)
    sorted_indices = np.argsort(norms)[::-1]
    sorted_vectors = chn_vcts[sorted_indices]
    sorted_norms = norms[sorted_indices]

    half = num_vectors // 2
    idx_large = slice(0, half)
    idx_small = slice(half, num_vectors)

    large_group = {
        'vectors': sorted_vectors[idx_large],
        'norms': sorted_norms[idx_large],
        'indices': sorted_indices[idx_large],
    }
    small_group = {
        'vectors': sorted_vectors[idx_small],
        'norms': sorted_norms[idx_small],
        'indices': sorted_indices[idx_small],
    }
    return {
        'large_group': large_group,
        'small_group': small_group,
    }


if __name__ == "__main__":
    chn_paras = channel_parameter(np.array([100, 200]), ["LOS", "NLOS"])
    print(chn_paras)
    chn_vcts = channel_vector(chn_paras)
    print(chn_vcts.shape)
    chn_group = channel_group(chn_vcts)
    print(chn_group)