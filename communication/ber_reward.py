"""
功能模块：根据 agent 网格位置计算 NOMA 分簇、功率分配与误码率，返回 BER 与 reward。
供 env、rl_algorithms 等调用，不包含测试或调试逻辑。
"""
import numpy as np

from communication.channel import (
    channel_parameter,
    channel_vector,
    channel_group,
    path_loss_batch,
)
from communication.diagonalization_precoding import matrix_cal
from communication.SIC import compute_sinr, epsilon, ber_to_reward, get_noma_powers
from communication.utils import dbm2watt
from config.yml_config import _get_parser


def _get_grid_size():
    """从环境参数获取网格物理尺寸（米），若无则使用默认 0.4"""
    try:
        from config.yml_config import _get_env_parser
        return float(_get_env_parser().parse_args().grid_size)
    except Exception:
        return 0.4


def get_ber_reward(agent_states, grid_size, antenna_position, get_los_nlos=None, P_max=None, verbose_sinr=False):
    """
    根据当前 agent 网格位置计算信道、分簇、功率分配与误码率，返回每个 agent 的 BER 及 reward。
    分簇：按信道向量范数分为大组/小组，大组第 k 个与小组第 k 个为一簇。
    功率：大组 P_max/2^p..P_max/2^(2p-1)，小组 P_max/2..P_max/2^p。
    :param agent_states: 列表，每个元素为 (grid_x, grid_y) 网格坐标
    :param grid_size: 网格对应的物理尺寸（米）
    :param antenna_position: 基站天线位置 (x, y)，与地图同一坐标尺度
    :param get_los_nlos: 可选，函数 (x, y) -> 'los'|'nlos'，用于每个格点；None 则全用 LOS
    :param P_max: 最大功率 (W)，None 则从 config 读取
    :param verbose_sinr: 是否打印每个簇的 SINR 计算过程
    :return: (rewards_ber, ber_per_agent, sinr_per_agent)，分别为 -log(ber) 列表、误码率列表、SINR 列表（线性值）
    """
    if P_max is None:
        P_max = _get_parser().parse_args().P_max
    num_agents = len(agent_states)
    if num_agents == 0:
        return [], [], []

    if get_los_nlos is not None:
        models = [get_los_nlos(int(s[0]), int(s[1])).upper() for s in agent_states]
    else:
        models = ["LOS"] * num_agents
    # 天线与位置统一为物理距离（米）：config 中 antenna_position 为网格坐标，需乘以 grid_size
    antenna_phys = np.asarray(antenna_position, dtype=np.float64).reshape(2) * grid_size
    positions = np.array([[float(s[0]) * grid_size, float(s[1]) * grid_size] for s in agent_states], dtype=np.float64)
    distances = np.linalg.norm(positions - antenna_phys, axis=1)
    distances = np.maximum(distances, 1e-3)
    chn_paras = channel_parameter(distances, models=models)
    thetas = np.pi / 2
    chn_vcts = channel_vector(chn_paras, thetas)
    groups = channel_group(chn_vcts)
    large_group = groups["large_group"]
    small_group = groups["small_group"]
    p = min(len(large_group["indices"]), len(small_group["indices"]))
    cluster_list = [
        np.vstack([large_group["vectors"][k], small_group["vectors"][k]])
        for k in range(p)
    ]
    large_powers, small_powers = get_noma_powers(P_max, p)

    ber_per_agent = np.ones(num_agents) * 0.5
    sinr_per_agent = np.full(num_agents, np.nan, dtype=np.float64)

    for m in range(p):
        H_m = cluster_list[m]
        w_m = matrix_cal(cluster_list, m)
        if w_m.size == 0 or w_m.shape[1] < 2:
            continue
        w = w_m[:, 0]
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            continue
        w = w / w_norm
        h_strong = large_group["vectors"][m]
        h_weak = small_group["vectors"][m]
        P1 = large_powers[m]
        P2 = small_powers[m]
        if verbose_sinr:
            print("\n--- 簇 {} (强用户 agent {}, 弱用户 agent {}) ---".format(
                m, large_group["indices"][m], small_group["indices"][m]))
        sinr_1, sinr_2 = compute_sinr(h_strong, h_weak, w, P1, P2, verbose=verbose_sinr)
        ber_per_agent[large_group["indices"][m]] = epsilon(sinr_1)
        ber_per_agent[small_group["indices"][m]] = epsilon(sinr_2)
        sinr_per_agent[large_group["indices"][m]] = sinr_1
        sinr_per_agent[small_group["indices"][m]] = sinr_2

    if num_agents % 2 != 0 and len(small_group["indices"]) > p:
        idx_extra = small_group["indices"][p]
        H_single = small_group["vectors"][p : p + 1]
        cluster_list_single = cluster_list + [H_single]
        w_single = matrix_cal(cluster_list_single, p)
        if w_single.size > 0 and np.linalg.norm(w_single) > 1e-12:
            w_single = w_single[:, 0] / np.linalg.norm(w_single[:, 0])
            g = np.abs(np.dot(H_single.ravel(), w_single.ravel())) ** 2
            sigma = dbm2watt(_get_parser().parse_args().power_AWGN)
            sigma = max(sigma, 1e-25)
            sinr = (P_max / 2) * g / sigma
            if verbose_sinr:
                print("\n--- 单用户 (agent {}) ---".format(idx_extra))
                print("  [SINR 计算] σ² = {:.6e} W,  g = |h^H w|² = {:.6e},  P = P_max/2 = {:.6e} W".format(sigma, g, P_max / 2))
                print("  [SINR 计算] SINR = P*g/σ² = {:.6e} = {:.2f} dB".format(sinr, 10 * np.log10(max(sinr, 1e-20))))
            ber_per_agent[idx_extra] = epsilon(sinr)
            sinr_per_agent[idx_extra] = sinr

    rewards_ber = [ber_to_reward(ber_per_agent[i]) for i in range(num_agents)]
    ber_list = [float(ber_per_agent[i]) for i in range(num_agents)]
    sinr_list = [float(sinr_per_agent[i]) for i in range(num_agents)]
    return rewards_ber, ber_list, sinr_list
