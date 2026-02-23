"""
训练入口：调用 rl_algorithms.train 子模块的 main()。
运行方式：在项目根目录执行 python -m rl_algorithms.train，或 python rl_algorithms/train.py。
"""
import sys
import os

# 直接运行本文件时，脚本所在目录在 sys.path[0]，需把项目根加入 path 才能 import rl_algorithms
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_algorithms.train import main

if __name__ == "__main__":
    main()
