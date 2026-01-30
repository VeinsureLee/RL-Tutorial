import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.param_arguments import parser
from .channel import *


number_of_antenna = parser.parse_args().number_of_antenna


def matrix_cal(cluster_list, m):
    """
    按图一 / 论文 (2-7)~(2-10) 计算第 m 簇的预编码矩阵 w_m：两次 SVD，w_m 位于干扰信道零空间。
    第一次 SVD：H_tilde_m 得 V_m^(0)（零空间）；第二次 SVD：等效信道 H_m @ V_m^(0) 得 V_m^(1)；
    最终 w_m = V_m^(0) @ V_m^(1)。
    :param cluster_list: list of ndarray
        每个元素为第 k 簇的信道矩阵，形状 (N_m, Nt)，通常 N_m=2（大组用户、小组用户）
    :param m: int，当前簇下标
    :return w_m: ndarray，形状 (Nt, N_m)，第 m 簇的预编码矩阵
    """
    cluster_list = [np.asarray(c, dtype=np.complex128) for c in cluster_list]
    H_m = cluster_list[m]
    Nt = H_m.shape[1]
    N_m = H_m.shape[0]

    H_tilde = []
    for i, cluster in enumerate(cluster_list):
        if i != m:
            c = np.asarray(cluster, dtype=np.complex128)
            if c.size > 0 and c.shape[0] > 0:
                H_tilde.append(c)

    if len(H_tilde) == 0:
        H_tilde = np.zeros((0, Nt), dtype=np.complex128)
    else:
        H_tilde = np.vstack(H_tilde)

    if H_tilde.shape[0] == 0:
        V0_tilde = np.eye(Nt, dtype=np.complex128)
    else:
        U_t, S_t, Vh_t = np.linalg.svd(H_tilde, full_matrices=True)
        V_t = Vh_t.conj().T
        rank = min(H_tilde.shape[0], np.sum(S_t > 1e-10 * S_t[0]))
        rank = int(np.clip(rank, 0, Nt))
        V0_tilde = V_t[:, rank:]

    H_eff = H_m @ V0_tilde
    if H_eff.size == 0:
        w_m = np.zeros((Nt, N_m), dtype=np.complex128)
        return w_m
    U_e, S_e, Vh_e = np.linalg.svd(H_eff, full_matrices=False)
    V1 = Vh_e.conj().T[:, :N_m]
    w_m = V0_tilde @ V1
    return w_m


def build_W_matrix(cluster_list):
    """
    根据簇列表计算完整预编码矩阵 W = [w_0, w_1, ..., w_{p-1}]，形状 (Nt, 2*p)。
    :param cluster_list: list of ndarray，每个形状 (2, Nt)
    :return W: ndarray，形状 (Nt, 2*p)
    """
    if not cluster_list:
        return np.zeros((number_of_antenna, 0), dtype=np.complex128)
    w_list = []
    for m in range(len(cluster_list)):
        w_m = matrix_cal(cluster_list, m)
        w_list.append(w_m)
    W = np.hstack(w_list)
    return W
