"""MADQN 预训练模型测试。"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_algorithms.test.run import test


def main():
    print("=" * 50)
    print("MADQN 预训练模型测试")
    print("=" * 50)
    test(algo="madqn")


if __name__ == "__main__":
    main()
