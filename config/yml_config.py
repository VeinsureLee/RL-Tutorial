"""
统一从 yml 加载配置，替代 config.params。供 env、communication、visualization、radio_map、generator 等使用。
"""
import os
import sys
from typing import Any, Union, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.config_handler import load_yml
from utils.path_tool import get_abs_path


def _get_yml_value(data: dict, key: str, default: Any = None) -> Any:
    """从 yml 结构 { key: { value: ..., description: ... } } 中取 value。"""
    if not data or key not in data:
        return default
    v = data[key]
    return v.get("value", default) if isinstance(v, dict) else v


def _load_base_yml(name: str) -> dict:
    """加载 config/base/{name}.yml。"""
    path = get_abs_path(os.path.join("config", "base", f"{name}.yml"))
    if not os.path.isfile(path):
        return {}
    out = load_yml(path)
    return out or {}


# ---------------------------------------------------------------------------
# 信道配置（channel.yml），兼容 parser.parse_args().xxx
# ---------------------------------------------------------------------------

class _ChannelArgs:
    """兼容原 parser.parse_args() 的只读对象。"""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
    def parse_args(self):
        return self


def get_channel_config() -> _ChannelArgs:
    """从 config/base/channel.yml 加载信道参数，返回兼容 parser.parse_args() 的对象。"""
    _channel_yml = _load_base_yml("channel")
    _map_yml = _load_base_yml("map")
    ap = _get_yml_value(_channel_yml, "antenna_position", None)
    if ap is None:
        ap = _get_yml_value(_map_yml, "antenna_position", [60, 30])
    ap = tuple(ap) if isinstance(ap, list) else ap

    power_awgn_dbm_hz = float(_get_yml_value(_channel_yml, "power_AWGN", -143.0))
    bw = float(_get_yml_value(_channel_yml, "channel_bandwidth", 1.0e7))
    # 噪声功率 (mW): N0(dBm/Hz) + 10*log10(BW) -> dBm -> mW
    noise_dbm = power_awgn_dbm_hz + 10 * np.log10(bw)
    noise_power_mw = 10 ** (noise_dbm / 10)  # dBm -> mW

    return _ChannelArgs({
        "carrier_frequency": float(_get_yml_value(_channel_yml, "carrier_frequency", 3.5)),
        "sigma_rayleigh": float(_get_yml_value(_channel_yml, "sigma_rayleigh", 1.2)),
        "number_of_antenna": int(_get_yml_value(_channel_yml, "number_of_antenna", 128)),
        "antenna_position": ap,
        "power_AWGN": power_awgn_dbm_hz,
        "channel_bandwidth": bw,
        "channel_block_length": int(_get_yml_value(_channel_yml, "channel_block_length", 256)),
        "packet_size": int(_get_yml_value(_channel_yml, "packet_size", 16)),
        "P_sum": float(_get_yml_value(_channel_yml, "P_sum", 100.0)),
        "P_min_diff": float(_get_yml_value(_channel_yml, "P_min_diff", 5.0)),
        "num_power_levels": int(_get_yml_value(_channel_yml, "num_power_levels", 3)),
        "noise_power_mw": noise_power_mw,
    })


# 模块级单例，兼容 from config.params import parser
parser = None
param_parser = None

def _get_parser():
    global parser, param_parser
    if parser is None:
        parser = get_channel_config()
        param_parser = parser
    return parser


# ---------------------------------------------------------------------------
# 环境配置（env.yml + dynamic 场景），兼容 env_parser.parse_args().xxx
# ---------------------------------------------------------------------------

class _EnvArgs:
    """兼容原 env_parser.parse_args() 的只读对象。"""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
    def parse_args(self):
        return self


def get_env_config() -> dict:
    """
    从 yml 加载环境所需全部配置（config/base + config/dynamic 场景）。
    若 config/base/map.yml 的 number_of_robots 与 config/dynamic/agent_num.yml 不一致
    或 agent_num.yml 不存在，则重新生成 config/dynamic 下各文件。
    :return: dict，包含 map_size, grid_size, start_states, target_states, forbidden_areas,
             action_space, reward_*, los_nlos_grid, map_config_map_size, antenna_position 等。
    """
    from config.generator.main import load_scenario_from_dynamic, get_or_create_map_and_agents
    from utils.config_handler import load_yml

    _env_yml = _load_base_yml("env")
    _map_yml = _load_base_yml("map")
    _channel_yml = _load_base_yml("channel")

    map_size = _get_yml_value(_map_yml, "map_size", [48, 24])
    map_size = tuple(map_size) if isinstance(map_size, list) else map_size
    map_size = np.array(map_size, dtype=np.float64)

    # agent num 判断：map.yml 的 number_of_robots 与 agent_num.yml 一致则直接加载，否则重新生成 dynamic
    map_num_agents = int(_get_yml_value(_map_yml, "number_of_robots", 4))
    dynamic_dir = get_abs_path("config/dynamic")
    agent_num_path = os.path.join(dynamic_dir, "agent_num.yml")
    stored_agent_num = None
    if os.path.isfile(agent_num_path):
        stored = load_yml(agent_num_path)
        if isinstance(stored, dict) and "num_agents" in stored:
            stored_agent_num = int(stored["num_agents"])
    if stored_agent_num is None or stored_agent_num != map_num_agents:
        base = get_base_map_and_seed()
        get_or_create_map_and_agents(
            random_seed=base["random_seed"],
            num_agents=map_num_agents,
            map_size=base["map_size"],
            antenna_position=base["antenna_position"],
            num_forbidden_squares=base["num_forbidden_squares"],
            square_size_range=base["square_size_range"],
            dynamic_dir=dynamic_dir,
            force_regenerate=True,
        )

    scenario = load_scenario_from_dynamic(
        dynamic_dir=dynamic_dir,
        map_size=map_size,
    )
    if scenario is None:
        raise FileNotFoundError(
            "config/dynamic 下缺少完整场景 yml（start_states, target_states, forbidden_areas, "
            "los_region, nlos_region, los_nlos_grid）。请先运行 config.generator.main 生成。"
        )

    action_directions_raw = _get_yml_value(
        _env_yml, "action_directions",
        [[0, 1], [1, 0], [0, -1], [-1, 0]],
    )
    action_directions = [tuple(a) for a in action_directions_raw] if action_directions_raw else [(0, 1), (1, 0), (0, -1), (-1, 0)]

    antenna_position = _get_yml_value(_channel_yml, "antenna_position", None)
    if antenna_position is None:
        antenna_position = _get_yml_value(_map_yml, "antenna_position", [60, 30])
    antenna_position = tuple(antenna_position) if isinstance(antenna_position, list) else antenna_position

    _channel_cfg = get_channel_config()
    _map_yml_heights = _load_base_yml("map")

    return {
        "map_size": map_size,
        "grid_size": float(_get_yml_value(_env_yml, "grid_size", 0.4)),
        "start_states": scenario["start_states"],
        "target_states": scenario["target_states"],
        "forbidden_areas": scenario["forbidden_areas"],
        "action_directions": action_directions,
        "reward_goal": float(_get_yml_value(_env_yml, "reward_goal", 10)),
        "reward_forbidden": float(_get_yml_value(_env_yml, "reward_forbidden", -5)),
        "reward_closer": float(_get_yml_value(_env_yml, "reward_closer", -0.8)),
        "reward_farther": float(_get_yml_value(_env_yml, "reward_farther", 0.1)),
        "reward_same": float(_get_yml_value(_env_yml, "reward_same", 0.0)),
        "omega": float(_get_yml_value(_env_yml, "omega", 1.0)),
        "los_nlos_grid": scenario["los_nlos_grid"],
        "antenna_position": antenna_position,
        # 高度参数
        "h_AP": float(_get_yml_value(_map_yml_heights, "h_AP", 2.0)),
        "h_robot": float(_get_yml_value(_map_yml_heights, "h_robot", 1.5)),
        "h_block": float(_get_yml_value(_map_yml_heights, "h_block", 3.0)),
        # 通信参数
        "n_antenna": _channel_cfg.number_of_antenna,
        "carrier_freq_ghz": _channel_cfg.carrier_frequency,
        "sigma_rayleigh": _channel_cfg.sigma_rayleigh,
        "P_sum": _channel_cfg.P_sum,
        "P_min_diff": _channel_cfg.P_min_diff,
        "num_power_levels": _channel_cfg.num_power_levels,
        "channel_block_length": _channel_cfg.channel_block_length,
        "packet_size": _channel_cfg.packet_size,
        "noise_power_mw": _channel_cfg.noise_power_mw,
    }


def get_env_parser() -> _EnvArgs:
    """返回兼容 env_parser.parse_args() 的对象。"""
    return _EnvArgs(get_env_config())


# 模块级单例
_env_parser = None

def _get_env_parser():
    global _env_parser
    if _env_parser is None:
        _env_parser = get_env_parser()
    return _env_parser


# ---------------------------------------------------------------------------
# 地图/场景：供 env.visualization、env.radio_map、rl_algorithms.test 等使用
# ---------------------------------------------------------------------------

def get_map_and_scenario():
    """返回 (map_size, forbidden_areas, get_los_nlos, antenna_position, scenario_dict)。"""
    from config.generator.main import load_scenario_from_dynamic
    from config.generator.region_generator import get_los_nlos as _get_los_nlos

    _map_yml = _load_base_yml("map")
    _channel_yml = _load_base_yml("channel")
    map_size = _get_yml_value(_map_yml, "map_size", [120, 60])
    map_size = tuple(map_size) if isinstance(map_size, list) else map_size
    map_size = np.array(map_size, dtype=np.float64)

    scenario = load_scenario_from_dynamic(
        dynamic_dir=get_abs_path("config/dynamic"),
        map_size=map_size,
    )
    if scenario is None:
        raise FileNotFoundError("config/dynamic 下缺少完整场景 yml，请先运行 config.generator.main 生成。")

    def get_los_nlos(x: int, y: int, los_nlos_grid_=None, map_size_=None):
        return _get_los_nlos(
            x, y,
            los_nlos_grid=los_nlos_grid_,
            map_size=map_size_,
            default_los_nlos_grid=scenario["los_nlos_grid"],
            default_map_size=tuple(int(x) for x in map_size),
        )

    ap = _get_yml_value(_channel_yml, "antenna_position", None)
    if ap is None:
        ap = _get_yml_value(_map_yml, "antenna_position", [60, 30])
    antenna_position = tuple(ap) if isinstance(ap, list) else ap

    return map_size, scenario["forbidden_areas"], get_los_nlos, antenna_position, scenario


# ---------------------------------------------------------------------------
# Generator 用：仅 base 参数（不触发 dynamic 场景加载）
# ---------------------------------------------------------------------------

def get_base_map_and_seed():
    """从 config/base 读取 map 与 random_seed，供 generator 生成场景。返回 dict。"""
    _map_yml = _load_base_yml("map")
    _seed_yml = _load_base_yml("random_seed")
    map_size = _get_yml_value(_map_yml, "map_size", [120, 60])
    map_size = tuple(map_size) if isinstance(map_size, list) else map_size
    antenna_position = _get_yml_value(_map_yml, "antenna_position", [60, 30])
    antenna_position = tuple(antenna_position) if isinstance(antenna_position, list) else antenna_position
    return {
        "map_size": map_size,
        "antenna_position": antenna_position,
        "num_agents": int(_get_yml_value(_map_yml, "number_of_robots", 4)),
        "num_forbidden_squares": int(_get_yml_value(_map_yml, "num_forbidden_squares", 5)),
        "square_size_range": tuple(_get_yml_value(_map_yml, "square_size_range", [7, 12])),
        "random_seed": int(_get_yml_value(_seed_yml, "random_seed", 42)),
    }


# ---------------------------------------------------------------------------
# RL 配置（rl.yml）
# ---------------------------------------------------------------------------

_RL_DEFAULTS = dict(
    algo="madqn", lr=1e-4, gamma=0.9,
    epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.99,
    num_iterations=5, num_episodes=50, episode_length=5000,
    batch_size=128, mini_batch_size=128, hidden_dim=128, update_freq=100,
    replay_buffer_size=50000,
    test_max_steps=500, model_dir="models",
)


def get_rl_config(**overrides) -> dict:
    """
    从 config/base/rl.yml 加载 RL 超参数，再用 overrides 覆盖。
    :param overrides: 调用方传入的参数，优先级最高
    :return: dict，包含所有 RL 超参数
    """
    _rl_yml = _load_base_yml("rl")
    cfg = {}
    for k, default in _RL_DEFAULTS.items():
        v = _get_yml_value(_rl_yml, k, default)
        cfg[k] = type(default)(v) if v is not None else default
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ---------------------------------------------------------------------------
# 兼容旧接口：按需加载并暴露与 params 同名的量
# ---------------------------------------------------------------------------

def get_map_size():
    """地图尺寸 (rows, cols)。"""
    _map_yml = _load_base_yml("map")
    ms = _get_yml_value(_map_yml, "map_size", [120, 60])
    return tuple(ms) if isinstance(ms, list) else ms


def get_antenna_position():
    """天线位置。"""
    return _get_parser().antenna_position


def get_los_nlos_global(x: int, y: int, los_nlos_grid_=None, map_size_=None):
    """根据离散坐标返回 LOS/NLOS，需已存在 dynamic 场景。"""
    _, _, getter, _, _ = get_map_and_scenario()
    return getter(x, y)
