"""单元测试：src/config/。"""
import os

import numpy as np
import pytest


# ---------------------------------------------------------------- yml 加载

def test_get_env_config_has_expected_keys(env_cfg):
    must_have = {
        "map_size", "grid_size",
        "start_states", "target_states", "forbidden_areas",
        "action_directions",
        "reward_goal", "reward_forbidden", "reward_closer", "reward_farther",
        "reward_same", "reward_step", "omega",
        "ber_reward_better", "ber_reward_worse",
        "antenna_position", "h_AP", "h_robot", "h_block",
        "n_antenna", "carrier_freq_ghz", "sigma_rayleigh",
        "P_sum", "num_power_levels",
        "channel_block_length", "packet_size", "noise_power_mw",
        "random_seed",
    }
    missing = must_have - set(env_cfg.keys())
    assert not missing, f"env_cfg 缺字段: {missing}"


def test_rl_cfg_defaults(rl_cfg):
    # 类型与范围检查
    assert rl_cfg["algo"] in {"dqn", "madqn"}
    assert 0 < rl_cfg["lr"] < 1
    assert 0 < rl_cfg["gamma"] <= 1
    assert rl_cfg["epsilon_min"] <= rl_cfg["epsilon"] <= 1
    assert rl_cfg["batch_size"] >= 1
    assert rl_cfg["hidden_dim"] >= 1


def test_noise_power_mw_value(env_cfg):
    """噪声功率换算：N0(dBm/Hz) + 10*log10(BW) -> dBm -> mW。"""
    # 用 channel.yml 默认值 (-143, 1e7) 反推：-73 dBm = 5.0119e-8 mW
    expected_dbm = -143.0 + 10 * np.log10(1.0e7)
    expected_mw = 10 ** (expected_dbm / 10)
    got = env_cfg["noise_power_mw"]
    assert np.isclose(got, expected_mw, rtol=1e-6), \
        f"noise_power_mw: got={got}, expected={expected_mw}"


# ---------------------------------------------------------------- scenario 重建

def test_scenario_shapes(env_cfg):
    """start_states / target_states 长度一致且均在地图范围内。"""
    rows, cols = env_cfg["map_size"]
    assert len(env_cfg["start_states"]) == len(env_cfg["target_states"])
    for r, c in env_cfg["start_states"]:
        assert 0 <= r < rows and 0 <= c < cols
    for r, c in env_cfg["target_states"]:
        assert 0 <= r < rows and 0 <= c < cols


def test_scenario_start_target_disjoint(env_cfg):
    """同一 agent 的起点与终点不重合。"""
    for (s, t) in zip(env_cfg["start_states"], env_cfg["target_states"]):
        assert tuple(s) != tuple(t)


def test_scenario_not_in_forbidden(env_cfg):
    """起点 / 终点都不在禁区方框内。"""
    def _forbidden_set(areas):
        s = set()
        for area in areas:
            pos, size = area
            r0, c0 = pos
            for dr in range(size):
                for dc in range(size):
                    s.add((r0 + dr, c0 + dc))
        return s

    fset = _forbidden_set(env_cfg["forbidden_areas"])
    for r, c in env_cfg["start_states"]:
        assert (r, c) not in fset
    for r, c in env_cfg["target_states"]:
        assert (r, c) not in fset


# ---------------------------------------------------------------- path_tool

def test_get_root_path_resolves_repo_root(repo_root):
    from utils.path_tool import get_root_path, get_abs_path
    assert os.path.samefile(get_root_path(), str(repo_root))
    assert os.path.samefile(
        get_abs_path("config/base"),
        str(repo_root / "config" / "base"),
    )


# ---------------------------------------------------------------- 障碍生成约束
# 新加的两个约束要单独验证，因为 _ensure_scenario 缓存命中时 env_cfg fixture 不会
# 触发重生成；这里直接调 generate_forbidden_areas 显式测。

def _bbox_chebyshev_to_point(r0, c0, sz, point):
    pr, pc = point
    dr = max(r0 - pr, pr - (r0 + sz - 1), 0)
    dc = max(c0 - pc, pc - (c0 + sz - 1), 0)
    return max(dr, dc)


def _bbox_pair_chebyshev(a, b):
    (ar0, ac0), asz = a
    (br0, bc0), bsz = b
    gap_r = max(ar0 - (br0 + bsz - 1), br0 - (ar0 + asz - 1), 0)
    gap_c = max(ac0 - (bc0 + bsz - 1), bc0 - (ac0 + asz - 1), 0)
    if gap_r == 0 and gap_c == 0:
        return 0
    return max(gap_r, gap_c)


def test_obstacles_respect_antenna_keepout():
    """配置 keepout=12 时，所有障碍 bbox 与天线 Chebyshev 距离 >= 12。"""
    from config.generator.forbidden_generator import generate_forbidden_areas
    map_size = (120, 60)
    antenna = (60, 30)
    keepout = 12
    areas = generate_forbidden_areas(
        map_size=map_size, antenna_position=antenna,
        num_forbidden_squares=7, square_size_range=(6, 10),
        random_seed=42,
        antenna_keepout_margin=keepout, min_obstacle_spacing=4,
    )
    assert len(areas) == 7
    for (pos, sz) in areas:
        d = _bbox_chebyshev_to_point(pos[0], pos[1], sz, antenna)
        assert d >= keepout, \
            f"obstacle bbox=({pos}, sz={sz}) only {d} cells from antenna (need >= {keepout})"


def test_obstacles_respect_min_spacing():
    """配置 spacing=4 时，所有障碍两两 Chebyshev 间距 >= 4。"""
    from config.generator.forbidden_generator import generate_forbidden_areas
    spacing = 4
    areas = generate_forbidden_areas(
        map_size=(120, 60), antenna_position=(60, 30),
        num_forbidden_squares=7, square_size_range=(6, 10),
        random_seed=42,
        antenna_keepout_margin=12, min_obstacle_spacing=spacing,
    )
    for i in range(len(areas)):
        for j in range(i + 1, len(areas)):
            gap = _bbox_pair_chebyshev(areas[i], areas[j])
            assert gap >= spacing, \
                f"obstacles {i} vs {j} only {gap} cells apart (need >= {spacing})"


def test_obstacles_legacy_behaviour_when_zero():
    """keepout=0 + spacing=0 时退化为旧行为：仅"不重叠 + 不直接覆盖天线"。"""
    from config.generator.forbidden_generator import generate_forbidden_areas
    antenna = (60, 30)
    areas = generate_forbidden_areas(
        map_size=(120, 60), antenna_position=antenna,
        num_forbidden_squares=5, square_size_range=(6, 10),
        random_seed=7,
        antenna_keepout_margin=0, min_obstacle_spacing=0,
    )
    # 不变量：没有 bbox 直接覆盖天线（distance==0 即覆盖；旧逻辑也禁止）
    for (pos, sz) in areas:
        assert _bbox_chebyshev_to_point(pos[0], pos[1], sz, antenna) >= 1


def test_scenario_npz_roundtrip_with_new_params(tmp_path):
    """带新参数生成的 scenario.npz 能被 load_scenario 正确反序列化。"""
    from config.generator.main import get_or_create_scenario, load_scenario
    out = get_or_create_scenario(
        random_seed=123, num_agents=2,
        map_size=(120, 60), antenna_position=(60, 30),
        num_forbidden_squares=4, square_size_range=(6, 10),
        antenna_keepout_margin=10, min_obstacle_spacing=3,
        dynamic_dir=str(tmp_path), force_regenerate=True,
    )
    loaded = load_scenario(dynamic_dir=str(tmp_path))
    assert loaded is not None
    assert loaded["num_agents"] == 2
    assert tuple(loaded["map_size"]) == (120, 60)
    assert len(loaded["forbidden_areas"]) == len(out["forbidden_areas"]) == 4
    # 反序列化后形状仍是 [((r, c), size), ...]
    for (pos, sz) in loaded["forbidden_areas"]:
        assert isinstance(pos, tuple) and len(pos) == 2
        assert isinstance(sz, int) and sz > 0
