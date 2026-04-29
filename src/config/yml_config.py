"""
配置加载入口。

层次：
    1) config/base/*.yml      : 用户手改的 scalar（地图、奖励、RL 超参、信道）。
    2) config/dynamic/scenario.npz : 生成器落盘的场景（禁区、起终点、agent 数、AP 位置）。

对外只暴露：
    get_env_config()      -> dict，供 env.MultiRobotEnv 使用
    get_rl_config(**ov)   -> dict，RL 超参（命令行可覆盖）
    get_channel_config()  -> 带 parse_args() 兼容接口的对象，供 communication 模块使用
    get_base_map_and_seed() -> dict，供 generator 使用
"""
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.config_handler import load_yml
from utils.path_tool import get_abs_path


# ---------------------------------------------------------------------------
# 基础工具：从 yml 读 scalar
# ---------------------------------------------------------------------------

def _get_yml_value(data: dict, key: str, default: Any = None) -> Any:
    """从 { key: { value: ..., description: ... } } 结构取 value。"""
    if not data or key not in data:
        return default
    v = data[key]
    return v.get("value", default) if isinstance(v, dict) else v


def _load_base_yml(name: str) -> dict:
    path = get_abs_path(os.path.join("config", "base", f"{name}.yml"))
    if not os.path.isfile(path):
        return {}
    return load_yml(path) or {}


# ---------------------------------------------------------------------------
# 信道配置：供 communication 子模块使用（兼容 parser.parse_args().xxx）
# ---------------------------------------------------------------------------

class _Args:
    """兼容旧 parser.parse_args() 的只读对象。"""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def parse_args(self):
        return self


def get_channel_config() -> _Args:
    """读取 config/base/channel.yml，计算噪声功率，返回可点访问的配置对象。"""
    ch = _load_base_yml("channel")
    mp = _load_base_yml("map")
    ap = _get_yml_value(ch, "antenna_position", None)
    if ap is None:
        ap = _get_yml_value(mp, "antenna_position", [60, 30])
    ap = tuple(ap) if isinstance(ap, list) else ap

    awgn_dbm_hz = float(_get_yml_value(ch, "power_AWGN", -143.0))
    bw = float(_get_yml_value(ch, "channel_bandwidth", 1.0e7))
    # 噪声功率: N0(dBm/Hz) + 10*log10(BW) -> dBm -> mW
    noise_dbm = awgn_dbm_hz + 10 * np.log10(bw)
    noise_power_mw = 10 ** (noise_dbm / 10)

    return _Args({
        "carrier_frequency": float(_get_yml_value(ch, "carrier_frequency", 3.5)),
        "sigma_rayleigh": float(_get_yml_value(ch, "sigma_rayleigh", 1.2)),
        "number_of_antenna": int(_get_yml_value(ch, "number_of_antenna", 128)),
        "antenna_position": ap,
        "power_AWGN": awgn_dbm_hz,
        "channel_bandwidth": bw,
        "channel_block_length": int(_get_yml_value(ch, "channel_block_length", 256)),
        "packet_size": int(_get_yml_value(ch, "packet_size", 16)),
        "P_sum": float(_get_yml_value(ch, "P_sum", 100.0)),
        "num_power_levels": int(_get_yml_value(ch, "num_power_levels", 3)),
        "noise_power_mw": noise_power_mw,
    })


# 兼容 communication.channel 的 _get_parser / _get_env_parser
_parser_cache = None


def _get_parser() -> _Args:
    global _parser_cache
    if _parser_cache is None:
        _parser_cache = get_channel_config()
    return _parser_cache


_env_parser_cache = None


def _get_env_parser() -> _Args:
    global _env_parser_cache
    if _env_parser_cache is None:
        env_cfg = get_env_config()
        _env_parser_cache = _Args({
            "grid_size": env_cfg["grid_size"],
            "map_config_map_size": tuple(env_cfg["map_size"]),
        })
    return _env_parser_cache


# ---------------------------------------------------------------------------
# map / random_seed / scenario（供 generator 与 env）
# ---------------------------------------------------------------------------

def get_base_map_and_seed() -> dict:
    """生成器用的基础参数（地图/AP/agent 数/障碍数量/随机种子）。"""
    mp = _load_base_yml("map")
    sd = _load_base_yml("random_seed")
    map_size = _get_yml_value(mp, "map_size", [120, 60])
    map_size = tuple(map_size) if isinstance(map_size, list) else map_size
    ap = _get_yml_value(mp, "antenna_position", [60, 30])
    ap = tuple(ap) if isinstance(ap, list) else ap
    return {
        "map_size": map_size,
        "antenna_position": ap,
        "num_agents": int(_get_yml_value(mp, "number_of_robots", 4)),
        "num_forbidden_squares": int(_get_yml_value(mp, "num_forbidden_squares", 5)),
        "square_size_range": tuple(_get_yml_value(mp, "square_size_range", [7, 12])),
        # 障碍分散性约束（缺省 0 = 旧行为，向后兼容）
        "antenna_keepout_margin": int(_get_yml_value(mp, "antenna_keepout_margin", 0)),
        "min_obstacle_spacing": int(_get_yml_value(mp, "min_obstacle_spacing", 0)),
        "random_seed": int(_get_yml_value(sd, "random_seed", 42)),
    }


def _ensure_scenario(base: dict) -> dict:
    """
    加载 config/dynamic/scenario.npz；若缺失 / map_size 或 agent 数不一致则重建。
    """
    from config.generator.main import get_or_create_scenario, load_scenario

    existing = load_scenario()
    if existing is None \
            or existing["num_agents"] != base["num_agents"] \
            or tuple(existing["map_size"]) != tuple(base["map_size"]):
        return get_or_create_scenario(
            random_seed=base["random_seed"],
            num_agents=base["num_agents"],
            map_size=base["map_size"],
            antenna_position=base["antenna_position"],
            num_forbidden_squares=base["num_forbidden_squares"],
            square_size_range=base["square_size_range"],
            antenna_keepout_margin=base.get("antenna_keepout_margin", 0),
            min_obstacle_spacing=base.get("min_obstacle_spacing", 0),
            force_regenerate=True,
        )
    return existing


def get_env_config() -> dict:
    """合并 env.yml + channel.yml + scenario.npz + map 基础参数，返回 env 所需 dict。"""
    env_yml = _load_base_yml("env")
    map_yml = _load_base_yml("map")
    base = get_base_map_and_seed()
    scenario = _ensure_scenario(base)

    action_directions_raw = _get_yml_value(
        env_yml, "action_directions",
        [[0, 1], [1, 0], [0, -1], [-1, 0]],
    )
    action_directions = [tuple(a) for a in action_directions_raw]

    channel_cfg = get_channel_config()

    ap = _get_yml_value(_load_base_yml("channel"), "antenna_position", None)
    if ap is None:
        ap = scenario["antenna_position"]
    ap = tuple(ap) if isinstance(ap, (list, tuple)) else ap

    return {
        "map_size": scenario["map_size"],
        "grid_size": float(_get_yml_value(env_yml, "grid_size", 0.4)),
        "start_states": scenario["start_states"],
        "target_states": scenario["target_states"],
        "forbidden_areas": scenario["forbidden_areas"],
        "action_directions": action_directions,
        # 导航奖励
        "reward_goal": float(_get_yml_value(env_yml, "reward_goal", 10.0)),
        "reward_forbidden": float(_get_yml_value(env_yml, "reward_forbidden", -5.0)),
        "reward_closer": float(_get_yml_value(env_yml, "reward_closer", 0.5)),
        "reward_farther": float(_get_yml_value(env_yml, "reward_farther", -0.5)),
        "reward_same": float(_get_yml_value(env_yml, "reward_same", 0.0)),
        "reward_step": float(_get_yml_value(env_yml, "reward_step", -1.0)),
        "omega": float(_get_yml_value(env_yml, "omega", 1.0)),
        "ber_reward_better": float(_get_yml_value(env_yml, "ber_reward_better", 0.5)),
        "ber_reward_worse": float(_get_yml_value(env_yml, "ber_reward_worse", -0.5)),
        # 天线与高度
        "antenna_position": ap,
        "h_AP": float(_get_yml_value(map_yml, "h_AP", 2.0)),
        "h_robot": float(_get_yml_value(map_yml, "h_robot", 1.5)),
        "h_block": float(_get_yml_value(map_yml, "h_block", 3.0)),
        # 通信
        "n_antenna": channel_cfg.number_of_antenna,
        "carrier_freq_ghz": channel_cfg.carrier_frequency,
        "sigma_rayleigh": channel_cfg.sigma_rayleigh,
        "P_sum": channel_cfg.P_sum,
        "num_power_levels": channel_cfg.num_power_levels,
        "channel_block_length": channel_cfg.channel_block_length,
        "packet_size": channel_cfg.packet_size,
        "noise_power_mw": channel_cfg.noise_power_mw,
        "comm_model": str(_get_yml_value(_load_base_yml("channel"), "comm_model", "noma")),
        "random_seed": base["random_seed"],
    }


# ---------------------------------------------------------------------------
# RL 超参
# ---------------------------------------------------------------------------

_RL_DEFAULTS = dict(
    algo="madqn", lr=1e-3, gamma=0.99,
    epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.95,
    num_iterations=5, num_episodes=50, episode_length=5000,
    batch_size=64, hidden_dim=128, update_freq=10,
    replay_buffer_size=50000,
    train_interval=1,
    test_max_steps=500, model_dir="models",
    # PPO/MAPPO 专用
    gae_lambda=0.95, clip_epsilon=0.2,
    entropy_coef=0.01, value_coef=0.5,
    num_epochs=10, ppo_epochs=10,  # 别名兼容
    update_interval=2048,
)


def get_rl_config(algo: str = None, **overrides) -> dict:
    """从 rl.yml 读默认，overrides 中非 None 的值覆盖之。

    对 PPO/MAPPO 自动使用 ppo_lr (3e-4) 作为学习率，除非用户手动覆盖。
    """
    rl_yml = _load_base_yml("rl")
    cfg = {}
    for k, default in _RL_DEFAULTS.items():
        v = _get_yml_value(rl_yml, k, default)
        cfg[k] = type(default)(v) if v is not None else default

    # PPO/MAPPO 特殊处理：默认使用 ppo_lr 而非通用 lr
    if algo in ("ppo", "mappo"):
        ppo_lr = _get_yml_value(rl_yml, "ppo_lr", 3.0e-4)
        cfg["lr"] = type(_RL_DEFAULTS["lr"])(ppo_lr) if ppo_lr is not None else _RL_DEFAULTS["lr"]

    # algo 参数覆盖
    if algo is not None:
        cfg["algo"] = algo

    # 用户覆盖优先级最高
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg
