import numpy as np
from config.arguments import parser
import math


def path_loss(distances, models):
    """
    General path loss model
    :param distances: the distance between robot and BS
    :param model: path loss model type
    :return: path loss
    """
    freq = parser.parse_args().carrier_frequency
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
    sigma = parser.parse_args().sigma_rayleigh
    samples = len(distances)
    fading_factors = np.random.rayleigh(size=samples, scale=sigma)
    fading_factors = np.clip(fading_factors, a_min=1e-10, a_max=None)
    path_loss_values = path_loss(distances, models)
    chn_paras = path_loss_values - 10 * np.log10(fading_factors)
    return chn_paras

def channel_vector_value(chn_paras):
    beta = 10**(-chn_paras/20)
    return beta

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
    chn_vct_values = channel_vector_value(chn_paras)
    print(chn_vct_values)