"""
内部调试模块：信道、对角化、SIC、随机 agent BER 等测试入口。
运行方式：在项目根目录执行 python -m communication 或 python communication/run_tests.py。
"""
import numpy as np

from communication.channel import (
    channel_parameter,
    channel_vector,
    channel_group,
    path_loss_batch,
)
from communication.diagonalization_precoding import matrix_cal
from communication.SIC import compute_sinr
from communication.ber_reward import get_ber_reward, _get_grid_size
from config.yml_config import _get_parser, get_map_and_scenario, _get_env_parser


def test_channel():
    distances = np.array([1, 10, 100])
    models_los = ["LOS", "LOS", "LOS"]
    models = ["LOS", "NLOS", "LOS"]
    models_nlos = ["NLOS", "NLOS", "NLOS"]
    pl_values = path_loss_batch(distances, models)
    pl_values_los = path_loss_batch(distances, models_los)
    pl_values_nlos = path_loss_batch(distances, models_nlos)
    print("Path Loss Values(dB):", pl_values)
    print("Path Loss Values LOS(dB):", pl_values_los)
    print("Path Loss Values NLOS(dB):", pl_values_nlos)
    chn_paras = channel_parameter(distances, models=models)
    print(chn_paras)
    thetas = [0, np.pi / 2, np.pi / 4]
    chn_vcts = channel_vector(chn_paras, thetas)
    print(len(chn_vcts[0]), len(chn_vcts))
    print(chn_vcts)
    groups = channel_group(chn_vcts)

    print("=== Large group ===")
    print("Prime index:", groups['large_group']['indices'])
    print("Norm:", groups['large_group']['norms'])
    print("Vector length:", len(groups['large_group']['vectors']))

    print("\n=== Small group ===")
    print("Prime index:", groups['small_group']['indices'])
    print("Norm:", groups['small_group']['norms'])
    print("Vector length:", len(groups['small_group']['vectors']))


def test_diag():
    num_users_per_cluster = 6
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

    lg, sg = groups['large_group'], groups['small_group']
    p = min(len(lg['indices']), len(sg['indices']))
    cluster_list = [np.vstack([lg['vectors'][k], sg['vectors'][k]]) for k in range(p)]

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
    num_users_per_cluster = 6
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

    lg, sg = groups['large_group'], groups['small_group']
    p = min(len(lg['indices']), len(sg['indices']))
    cluster_list = [np.vstack([lg['vectors'][k], sg['vectors'][k]]) for k in range(p)]

    for m, H_m in enumerate(cluster_list):
        if H_m.size == 0:
            print(f"\nCluster {m+1}: empty cluster, skipping")
            continue

        print(f"\n========== Cluster {m + 1} ==========")

        w_m = matrix_cal(cluster_list, m)

        if w_m.shape[1] == 0:
            print(f"Cluster {m+1}: BD failed (no null space), skipping SINR")
            continue

        for k, H_k in enumerate(cluster_list):
            interf = np.linalg.norm(H_k @ w_m, ord='fro')
            print(f"|| H_{k} @ w_{m} ||_F = {interf:.3e}")

        w = w_m[:, 0]
        w = w / np.linalg.norm(w)

        eff_gains = np.abs(H_m @ w)**2
        sorted_idx = np.argsort(eff_gains)[::-1]
        h_strong = H_m[sorted_idx[0], :]
        h_weak   = H_m[sorted_idx[1], :]

        sinr_1, sinr_2 = compute_sinr(
            h_strong,
            h_weak,
            w,
            P1=1.0,
            P2=100.0
        )

        print(f"SINR strong user : {10 * np.log10(sinr_1):.2f} dB")
        print(f"SINR weak user   : {10 * np.log10(sinr_2):.2f} dB")


def test_ber_random_agents(seed=None):
    """
    测试：按 num_agent（number_of_robots）随机生成网格点，按分配规则分配功率并计算误码率。
    """
    if seed is not None:
        np.random.seed(seed)
    args = _get_parser().parse_args()
    num_agents = args.number_of_robots
    antenna_position = np.asarray(args.antenna_position, dtype=np.float64).reshape(2)
    P_max = args.P_max
    grid_size = _get_grid_size()
    _, _, map_get_los_nlos, _, _ = get_map_and_scenario()
    map_config_map_size = _get_env_parser().parse_args().map_config_map_size
    rows, cols = map_config_map_size[0], map_config_map_size[1]

    all_cells = [(i, j) for i in range(rows) for j in range(cols)]
    if len(all_cells) < num_agents:
        raise ValueError(
            f"地图格点数 {len(all_cells)} 小于 agent 数 {num_agents}，请增大地图或减少 number_of_robots"
        )
    chosen = np.random.choice(len(all_cells), size=num_agents, replace=False)
    agent_states = [all_cells[k] for k in chosen]

    rewards_ber, ber_list, sinr_list = get_ber_reward(
        agent_states,
        grid_size=grid_size,
        antenna_position=antenna_position,
        get_los_nlos=map_get_los_nlos,
        P_max=P_max,
        verbose_sinr=True,
    )

    print("========== 随机 Agent 误码率 / SINR 测试 ==========")
    print(f"Agent 数量: {num_agents}, 网格尺寸: {rows}x{cols}, grid_size={grid_size}m, P_max={P_max}W")
    print(f"天线位置 (网格坐标): ({antenna_position[0]:.0f}, {antenna_position[1]:.0f})")
    print("\n各 Agent 网格位置、SINR 与误码率:")
    for i in range(num_agents):
        sinr_val = sinr_list[i]
        sinr_db = 10 * np.log10(sinr_val) if sinr_val is not None and sinr_val > 0 and np.isfinite(sinr_val) else float('nan')
        sinr_str = f"SINR = {sinr_val:.4f} ({sinr_db:.2f} dB)" if np.isfinite(sinr_db) else "SINR = N/A"
        print(f"  Agent {i}: 网格 ({agent_states[i][0]}, {agent_states[i][1]})  {sinr_str}  BER = {ber_list[i]:.6e}  reward = {rewards_ber[i]:.4f}")
    avg_ber = float(np.nanmean(ber_list))
    valid_sinr = [s for s in sinr_list if s is not None and np.isfinite(s) and s > 0]
    avg_sinr = float(np.mean(valid_sinr)) if valid_sinr else float('nan')
    print(f"\n平均误码率: {avg_ber:.6e}")
    if np.isfinite(avg_sinr):
        print(f"平均 SINR: {avg_sinr:.4f} ({10 * np.log10(avg_sinr):.2f} dB)")
    else:
        print("平均 SINR: N/A")
    print("==================================================")
    return agent_states, ber_list, rewards_ber, sinr_list


def main():
    """默认运行随机 agent BER 测试；可改为 test_channel() / test_diag() / test_sic()。"""
    # test_channel()
    # test_diag()
    # test_sic()
    test_ber_random_agents(seed=42)


if __name__ == "__main__":
    main()
