import sys
import os
import numpy as np
from channel import *
from diagonalization_precoding import *
from SIC import *

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.env import Env
from rl_algorithms.agent import Agent


def test_channel():
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
    print("=== Cluster 1 ===")
    print("Prime index:", groups['cluster1']['indices'])
    print("Norm:", groups['cluster1']['norms'])
    print("Vector length:", len(groups['cluster1']['vectors']))

    print("\n=== Cluster 2 ===")
    print("Prime index:", groups['cluster2']['indices'])
    print("Norm:", groups['cluster2']['norms'])
    print("Vector length:", len(groups['cluster1']['vectors']))

    print("\n=== Cluster 3 ===")
    if groups['cluster3']:
        print("Prime index:", groups['cluster3']['indices'])
        print("Norm:", groups['cluster3']['norms'])
        print("Vector length:", len(groups['cluster1']['vectors']))


def test_diag():
    # ========== 参数 ==========
    num_users_per_cluster = 6  # 每簇用户数（原来太小）
    num_clusters = 3

    distances = np.concatenate([
        np.full(num_users_per_cluster, 10),
        np.full(num_users_per_cluster, 50),
        np.full(num_users_per_cluster, 100),
    ])

    models = (
            ["LOS"] * num_users_per_cluster +
            ["NLOS"] * num_users_per_cluster +
            ["LOS"] * num_users_per_cluster
    )

    thetas = np.random.uniform(0, np.pi, size=len(distances))
    chn_paras = channel_parameter(distances, models=models)
    chn_vcts = channel_vector(chn_paras, thetas)
    groups = channel_group(chn_vcts)

    cluster_list = [
        groups['cluster1']['vectors'],
        groups['cluster2']['vectors'],
        groups['cluster3']['vectors']
    ]
    Nt = number_of_antenna

    for m in range(len(cluster_list)):
        print(f"\n=== Cluster {m + 1} ===")
        w_m = matrix_cal(cluster_list, m)

        for k, H_k in enumerate(cluster_list):
            interference = np.linalg.norm(H_k @ w_m, ord='fro')
            print(f"|| H_{k} @ w_{m} ||_F = {interference:.3e}")


def test_sic():
    """
    完整 BD + NOMA + SIC 测试
    1️⃣ 按物理簇做 Block Diagonalization (簇间干扰消除)
    2️⃣ 簇内按信道增益自动划分强/弱用户
    3️⃣ 自动跳过空簇或 BD 失败簇
    4️⃣ 打印簇间干扰和簇内强/弱用户 SINR
    """
    # ========== 参数 ==========
    num_users_per_cluster = 6
    num_clusters = 3

    # 距离和信道模型
    distances = np.concatenate([
        np.full(num_users_per_cluster, 10),
        np.full(num_users_per_cluster, 50),
        np.full(num_users_per_cluster, 100),
    ])

    models = (
        ["LOS"] * num_users_per_cluster +
        ["NLOS"] * num_users_per_cluster +
        ["LOS"] * num_users_per_cluster
    )

    thetas = np.random.uniform(0, np.pi, size=len(distances))

    # ---------- 信道生成 ----------
    chn_paras = channel_parameter(distances, models=models)
    chn_vcts = channel_vector(chn_paras, thetas)
    groups = channel_group(chn_vcts)

    # 物理簇列表
    cluster_list = [
        groups['cluster1']['vectors'],
        groups['cluster2']['vectors'],
        groups['cluster3']['vectors']
    ]

    # ---------- BD + NOMA + SIC ----------
    for m, H_m in enumerate(cluster_list):
        if H_m.size == 0:
            print(f"\nCluster {m+1}: empty cluster, skipping")
            continue

        print(f"\n========== Cluster {m + 1} ==========")

        # 1️⃣ BD 计算
        w_m = matrix_cal(cluster_list, m)

        if w_m.shape[1] == 0:
            print(f"Cluster {m+1}: BD failed (no null space), skipping SINR")
            continue

        # 2️⃣ 簇间干扰打印
        for k, H_k in enumerate(cluster_list):
            interf = np.linalg.norm(H_k @ w_m, ord='fro')
            print(f"|| H_{k} @ w_{m} ||_F = {interf:.3e}")

        # 3️⃣ 簇内 NOMA 波束选择（第一根波束）
        w = w_m[:, 0]
        w = w / np.linalg.norm(w)

        # 4️⃣ 簇内强/弱用户划分
        eff_gains = np.abs(H_m @ w)**2
        sorted_idx = np.argsort(eff_gains)[::-1]  # 从大到小
        h_strong = H_m[sorted_idx[0], :]
        h_weak   = H_m[sorted_idx[1], :]

        # 5️⃣ SINR 计算
        sinr_1, sinr_2 = compute_sinr(
            h_strong,
            h_weak,
            w,
            P1=1.0,  # 强用户功率
            P2=100.0   # 弱用户功率
        )

        print(f"SINR strong user : {10 * np.log10(sinr_1):.2f} dB")
        print(f"SINR weak user   : {10 * np.log10(sinr_2):.2f} dB")




if __name__ == '__main__':
    # test_channel()
    test_sic()