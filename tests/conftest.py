"""
pytest 会在运行任何 test_*.py 之前自动加载 conftest.py。

这里负责两件事：
1. 把 ``src/`` 注入到 ``sys.path``，让测试里的 ``from config.xxx`` / ``from env.xxx``
   等 import 与 ``main.py`` 运行时保持一致。
2. 固定随机种子（numpy / torch 若有）以保证测试可复现。
"""
import os
import sys
from pathlib import Path

# 项目根 = tests/ 的上一级
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
np.random.seed(0)

try:
    import torch
    torch.manual_seed(0)
except ImportError:  # torch 不是硬依赖
    pass


import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """项目根的 Path。"""
    return _ROOT


@pytest.fixture(scope="session")
def env_cfg():
    """共享的 env_cfg 字典（仅 session 一次）。"""
    from config.yml_config import get_env_config
    return get_env_config()


@pytest.fixture(scope="session")
def rl_cfg():
    from config.yml_config import get_rl_config
    return get_rl_config()
