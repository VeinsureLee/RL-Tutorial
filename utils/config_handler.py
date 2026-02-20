"""
加载 config 目录下的 yml 配置，并支持输出参数设置。
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from utils.path_tool import get_abs_path


def _load_yml(config_path: str, encoding: str = "utf-8"):
    """通用 yml 加载。"""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def load_yml(config_path: str, encoding: str = "utf-8"):
    """
    通用 yml 加载，供其他模块（如 config）统一读取 yml 配置。
    与 _load_yml 一致。
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def save_yml(config_path: str, data, encoding: str = "utf-8", **yaml_kw) -> None:
    """
    通用 yml 写入，供其他模块统一写入 yml 配置。
    """
    d = os.path.dirname(config_path)
    if d:
        os.makedirs(d, exist_ok=True)
    kwargs = {"allow_unicode": True, "default_flow_style": False, "sort_keys": False}
    kwargs.update(yaml_kw)
    with open(config_path, "w", encoding=encoding) as f:
        yaml.dump(data, f, **kwargs)


def load_channel_config(
    config_path: str = None, encoding: str = "utf-8"
):
    """加载通信信道配置 (config/channel.yml)。"""
    path = config_path or get_abs_path("config/channel.yml")
    return _load_yml(path, encoding)


def load_map_config(config_path: str = None, encoding: str = "utf-8"):
    """加载地图与禁区配置 (config/map.yml)。"""
    path = config_path or get_abs_path("config/map.yml")
    return _load_yml(path, encoding)


def load_env_config(config_path: str = None, encoding: str = "utf-8"):
    """加载环境配置 (config/env.yml)。"""
    path = config_path or get_abs_path("config/env.yml")
    return _load_yml(path, encoding)


def load_agent_config(
    config_path: str = None, encoding: str = "utf-8"
):
    """加载智能体配置 (config/agent.yml)。"""
    path = config_path or get_abs_path("config/agent.yml")
    return _load_yml(path, encoding)

def load_random_seed_config(config_path: str = None, encoding: str = "utf-8"):
    """加载随机种子配置 (config/random_seed.yml)。"""
    path = config_path or get_abs_path("config/base/random_seed.yml")
    return _load_yml(path, encoding)

def _format_value(v):
    """将 value 转为可读字符串。"""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return str(v)
    return str(v)


def _print_section(name: str, data: dict, indent: str = "  "):
    """打印一个配置块（含 description 与各参数的 value/description）。"""
    if not data:
        return
    desc = data.get("description")
    if desc:
        print(f"{indent}description: {desc}")
    for k, v in data.items():
        if k == "description":
            continue
        if not isinstance(v, dict):
            print(f"{indent}{k}: {_format_value(v)}")
            continue
        sub_val = v.get("value")
        sub_desc = v.get("description", "")
        val_str = _format_value(sub_val) if "value" in v else "(由代码或运行时决定)"
        print(f"{indent}{k}:")
        print(f"{indent}  value: {val_str}")
        if sub_desc:
            print(f"{indent}  description: {sub_desc}")


def print_params_settings(
    channel_path: str = None,
    map_path: str = None,
    env_path: str = None,
    agent_path: str = None,
    random_seed_path: str = None,
    encoding: str = "utf-8",
):
    """
    加载 channel / map / env / agent 四个 yml，并将参数设置打印到控制台。
    """
    sections = [
        ("通信信道 (channel.yml)", load_channel_config, channel_path or get_abs_path("config/base/channel.yml")),
        ("地图与禁区 (map.yml)", load_map_config, map_path or get_abs_path("config/base/map.yml")),
        ("环境 (env.yml)", load_env_config, env_path or get_abs_path("config/base/env.yml")),
        ("随机种子 (random_seed.yml)", load_random_seed_config, random_seed_path or get_abs_path("config/base/random_seed.yml")),
    ]
    for title, loader, path in sections:
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
        try:
            data = loader(config_path=path, encoding=encoding)
            if data:
                _print_section(title, data)
            else:
                print("  (空配置)")
        except FileNotFoundError:
            print(f"  文件不存在: {path}")
        except Exception as e:
            print(f"  加载失败: {e}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print_params_settings()
