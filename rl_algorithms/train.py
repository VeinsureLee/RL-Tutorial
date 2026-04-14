"""
训练入口脚本。
运行方式：python rl_algorithms/train.py 或 python -m rl_algorithms.train
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_algorithms.train import main

if __name__ == "__main__":
    main()
