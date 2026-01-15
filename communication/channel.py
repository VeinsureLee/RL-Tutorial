import numpy as np
from config.param_arguments import parser
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
    Group channel vectors in descending order of their norms.
    - cluster1: The half with the largest norms
    - cluster2: The half with the smallest norms
    - cluster3: The middle one when the total number is odd (empty array otherwise)
    """
    chn_vcts = np.asarray(chn_vcts, dtype=np.complex128)

    num_vectors = chn_vcts.shape[0]

    norms = np.linalg.norm(chn_vcts, axis=1)

    sorted_indices = np.argsort(norms)[::-1]
    sorted_vectors = chn_vcts[sorted_indices]
    sorted_norms = norms[sorted_indices]

    half = num_vectors // 2

    if num_vectors % 2 == 0:
        idx1 = slice(0, half)
        idx2 = slice(half, num_vectors)
        idx3 = slice(0, 0)
    else:
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

