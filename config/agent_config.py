
"""
智能体起始与目标状态由 config.utils 根据 random_seed 与 number_of_robots
在 map_config 加载时一并生成或从 config/dynamic/agent.yml 读取，此处仅做兼容导出。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.map_config import start_states, target_states

__all__ = [
    "start_states",
    "target_states",
]
