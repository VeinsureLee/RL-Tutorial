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
        ap = _get_yml_value(_map_yml, "antenna_position", [24, 12])
    ap = tuple(ap) if isinstance(ap, list) else ap
    return _ChannelArgs({
        "carrier_frequency": float(_get_yml_value(_channel_yml, "carrier_frequency", 3.5)),
        "sigma_rayleigh": float(_get_yml_value(_channel_yml, "sigma_rayleigh", 1.2)),
        "number_of_antenna": int(_get_yml_value(_channel_yml, "number_of_antenna", 128)),
        "antenna_position": ap,
        "power_AWGN": float(_get_yml_value(_channel_yml, "power_AWGN", -143.0)),
        "channel_block_length": int(_get_yml_value(_channel_yml, "channel_block_length", 256)),
        "packet_size": int(_get_yml_value(_channel_yml, "packet_size", 16)),
        "number_of_robots": int(_get_yml_value(_channel_yml, "number_of_robots", 4)),
        "total_power": float(_get_yml_value(_channel_yml, "total_power", 1.0)),
        "rho_min": float(_get_yml_value(_channel_yml, "rho_min", 0.01)),
        "P_max": float(_get_yml_value(_channel_yml, "P_max", 100.0)),
        "P_min": float(_get_yml_value(_channel_yml, "P_min", 5.0)),
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

    action_space_raw = _get_yml_value(
        _env_yml, "action_space",
        [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]],
    )
    action_space = [tuple(a) for a in action_space_raw] if action_space_raw else [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]

    antenna_position = _get_yml_value(_channel_yml, "antenna_position", None)
    if antenna_position is None:
        antenna_position = _get_yml_value(_map_yml, "antenna_position", [24, 12])
    antenna_position = tuple(antenna_position) if isinstance(antenna_position, list) else antenna_position

    return {
        "map_size": map_size,
        "grid_size": float(_get_yml_value(_env_yml, "grid_size", 0.4)),
        "start_states": scenario["start_states"],
        "target_states": scenario["target_states"],
        "target_state": scenario["target_states"],
        "forbidden_areas": scenario["forbidden_areas"],
        "action_space": action_space,
        "reward_target": float(_get_yml_value(_env_yml, "reward_target", 10)),
        "reward_forbidden": float(_get_yml_value(_env_yml, "reward_forbidden", -5)),
        "reward_step": float(_get_yml_value(_env_yml, "reward_step", -1)),
        "reward_closer_to_target": float(_get_yml_value(_env_yml, "reward_closer_to_target", 1)),
        "los_nlos_grid": scenario["los_nlos_grid"],
        "map_config_map_size": tuple(int(x) for x in map_size),
        "antenna_position": antenna_position,
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
    map_size = _get_yml_value(_map_yml, "map_size", [48, 24])
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
        ap = _get_yml_value(_map_yml, "antenna_position", [24, 12])
    antenna_position = tuple(ap) if isinstance(ap, list) else ap

    return map_size, scenario["forbidden_areas"], get_los_nlos, antenna_position, scenario


# ---------------------------------------------------------------------------
# Generator 用：仅 base 参数（不触发 dynamic 场景加载）
# ---------------------------------------------------------------------------

def get_base_map_and_seed():
    """从 config/base 读取 map 与 random_seed，供 generator 生成场景。返回 dict。"""
    _map_yml = _load_base_yml("map")
    _seed_yml = _load_base_yml("random_seed")
    map_size = _get_yml_value(_map_yml, "map_size", [48, 24])
    map_size = tuple(map_size) if isinstance(map_size, list) else map_size
    antenna_position = _get_yml_value(_map_yml, "antenna_position", [24, 12])
    antenna_position = tuple(antenna_position) if isinstance(antenna_position, list) else antenna_position
    return {
        "map_size": map_size,
        "antenna_position": antenna_position,
        "num_agents": int(_get_yml_value(_map_yml, "number_of_robots", 4)),
        "num_forbidden_squares": int(_get_yml_value(_map_yml, "num_forbidden_squares", 5)),
        "square_size_range": tuple(_get_yml_value(_map_yml, "square_size_range", [3, 5])),
        "random_seed": int(_get_yml_value(_seed_yml, "random_seed", 42)),
    }


# ---------------------------------------------------------------------------
# 兼容旧接口：按需加载并暴露与 params 同名的量
# ---------------------------------------------------------------------------

def get_map_size():
    """地图尺寸 (rows, cols)。"""
    _map_yml = _load_base_yml("map")
    ms = _get_yml_value(_map_yml, "map_size", [48, 24])
    return tuple(ms) if isinstance(ms, list) else ms


def get_antenna_position():
    """天线位置。"""
    return _get_parser().antenna_position


def get_los_nlos_global(x: int, y: int, los_nlos_grid_=None, map_size_=None):
    """根据离散坐标返回 LOS/NLOS，需已存在 dynamic 场景。"""
    _, _, getter, _, _ = get_map_and_scenario()
    return getter(x, y)
