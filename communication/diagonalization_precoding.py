import numpy as np
from config.param_arguments import parser
from channel import *


number_of_antenna = parser.parse_args().number_of_antenna


def matrix_cal(cluster_list, m):
    """
    Calculate the precoding matrix w_m for the m-th cluster according to equations (2-7)~(2-10) in the paper.
    :param cluster_list: list
        Each element is a cluster returned by the channel_group function,
        where cluster['vectors'] has a shape of (N_m, Nt)

    :param m: int
        Index of the current cluster

    :return w_m : ndarray
        Precoding matrix with shape (Nt, N_m)
    """

    H_tilde = []
    for i, cluster in enumerate(cluster_list):
        if i != m:
            H_tilde.append(cluster)

    H_tilde = np.vstack(H_tilde)

    U_t, S_t, Vh_t = np.linalg.svd(H_tilde, full_matrices=True)
    V_t = Vh_t.conj().T

    rank = np.linalg.matrix_rank(H_tilde)
    V0_tilde = V_t[:, rank:]

    H_m = cluster_list[m]  # shape: (N_m, Nt)
    H_eff = H_m @ V0_tilde  # shape: (N_m, dim_null)

    U_e, S_e, Vh_e = np.linalg.svd(H_eff, full_matrices=False)
    V1 = Vh_e.conj().T[:, :H_m.shape[0]]  # V_m^{(1)}

    w_m = V0_tilde @ V1  # shape: (Nt, N_m)

    return w_m
