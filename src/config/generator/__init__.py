"""场景生成器：对外只暴露 scenario 构建/加载接口。"""
from config.generator.main import (
    get_or_create_scenario,
    load_scenario,
    save_scenario,
    SCENARIO_FILENAME,
)

__all__ = [
    "get_or_create_scenario",
    "load_scenario",
    "save_scenario",
    "SCENARIO_FILENAME",
]
