import numpy as np
from config.arguments import parser
import math


sigma = parser.parse_args().sigma_rayleigh
freq = parser.parse_args().carrier_frequency
number_of_antenna = parser.parse_args().number_of_antenna


def path_loss(distances, models):
    """
    General path loss model
    :param distances: the distance between robot and BS
    :param models: path loss model type
    :return: path loss
    """
    pl_list = []
    for model_idx, model in enumerate(models):
        if model == "LOS":
            pl_list.append(31.87 + 21.50 * np.log10(distances[model_idx]) + 19.0 * np.log10(freq))
        elif model == "NLOS":
            pl_los = 31.87 + 21.50 * np.log10(distances[model_idx]) + 19.0 * np.log10(freq)
            pl_sl = 33 + 25.50 * np.log10(distances[model_idx]) + 20 * np.log10(freq)
            pl_list.append(max(pl_los, pl_sl))
        else:
            raise ValueError("Unsupported path loss model: {}".format(model))
    return pl_list

def channel_parameter(distances, models):
    samples = len(distances)
    fading_factors = np.random.rayleigh(size=samples, scale=sigma)
    fading_factors = np.clip(fading_factors, a_min=1e-10, a_max=None)
    path_loss_values = path_loss(distances, models)
    chn_paras = path_loss_values - 10 * np.log10(fading_factors)
    return chn_paras

def channel_vector(chn_paras, thetas):
    beta = 10 ** (-chn_paras / 20)
    n = np.arange(number_of_antenna)[np.newaxis, :]
    phase_base = -1j * np.pi * np.sin(thetas)[:, np.newaxis]
    alpha_transmit = np.exp(n * phase_base) / np.sqrt(number_of_antenna)
    chn_vcts = beta[:, np.newaxis] * alpha_transmit
    return chn_vcts

def channel_group(chn_vcts):
    """
    按信道向量模长从大到小分组
    cluster1: 最大的一半
    cluster2: 最小的一半
    cluster3: 奇数时的中间那个（否则为空数组）
    """
    chn_vcts = np.asarray(chn_vcts, dtype=np.complex128)

    if chn_vcts.ndim != 2:
        raise ValueError("chn_vcts 必须是二维数组 [num_links, num_antennas]")

    num_vectors = chn_vcts.shape[0]
    if num_vectors == 0:
        raise ValueError("输入的信道向量集合不能为空")

    norms = np.linalg.norm(chn_vcts, axis=1)

    sorted_indices = np.argsort(norms)[::-1]
    sorted_vectors = chn_vcts[sorted_indices]
    sorted_norms = norms[sorted_indices]

    half = num_vectors // 2

    if num_vectors % 2 == 0:
        # 偶数
        idx1 = slice(0, half)
        idx2 = slice(half, num_vectors)
        idx3 = slice(0, 0)  # 空
    else:
        # 奇数
        idx1 = slice(0, half)
        idx3 = slice(half, half + 1)
        idx2 = slice(half + 1, num_vectors)

    cluster1 = {
        'vectors': sorted_vectors[idx1],
        'norms': sorted_norms[idx1],
        'indices': sorted_indices[idx1]
    }

    cluster2 = {
        'vectors': sorted_vectors[idx2],
        'norms': sorted_norms[idx2],
        'indices': sorted_indices[idx2]
    }

    cluster3 = {
        'vectors': sorted_vectors[idx3],
        'norms': sorted_norms[idx3],
        'indices': sorted_indices[idx3]
    }

    return {
        'cluster1': cluster1,
        'cluster2': cluster2,
        'cluster3': cluster3
    }


if __name__ == "__main__":
    distances = np.array([1, 10, 100])
    models_los = ["LOS", "LOS", "LOS"]
    models = ["LOS", "NLOS", "LOS"]
    models_nlos = ["NLOS", "NLOS", "NLOS"]
    pl_values = path_loss(distances, models=models)
    pl_values_los = path_loss(distances, models=models_los)
    pl_values_nlos = path_loss(distances, models=models_nlos)
    print("Path Loss Values(dB):", pl_values)
    print("Path Loss Values LOS(dB):", pl_values_los)
    print("Path Loss Values NLOS(dB):", pl_values_nlos)
    chn_paras = channel_parameter(distances, models=models)
    print(chn_paras)
    thetas = [0, np.pi / 2, np.pi / 4]
    chn_vcts = channel_vector(chn_paras, thetas)
    print(len(chn_vcts[0]),len(chn_vcts))
    print(chn_vcts)
    groups = channel_group(chn_vcts)

    # 打印结果
    print("=== 第一簇（最大的一半）===")
    print("原始索引:", groups['cluster1']['indices'])
    print("模长:", groups['cluster1']['norms'])

    print("\n=== 第二簇（最小的一半）===")
    print("原始索引:", groups['cluster2']['indices'])
    print("模长:", groups['cluster2']['norms'])

    print("\n=== 落单簇（奇数时存在）===")
    if groups['cluster3']:
        print("原始索引:", groups['cluster3']['indices'])
        print("模长:", groups['cluster3']['norms'])