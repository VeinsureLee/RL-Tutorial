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
