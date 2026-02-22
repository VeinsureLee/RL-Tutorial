"""
主入口：选择模型（dqn / madqn）进行训练或测试。
用法: python main.py --model dqn --mode train
      python main.py --model madqn --mode test
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from env.env import Env
from rl_algorithms.structure.dqn import DQN
from rl_algorithms.structure.madqn import MADQN
from rl_algorithms.train.train_dqn import train_dqn
from rl_algorithms.train.train_madqn import train_madqn

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_ROOT, "models")


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_env():
    return Env()


def _create_and_train_dqn(env, device):
    dqn = DQN(env, agent_id=0, lr=1e-3, gamma=0.99, iteration=5,
              epsilon=0.8, epsilon_decay=0.9, epsilon_min=0.1,
              num_episodes=50, episode_length=35000, mini_batch_size=64,
              update_freq=10, device=device)
    dqn, _ = train_dqn(env, dqn)
    path = os.path.join(MODEL_DIR, "dqn_model.pth")
    dqn.save(path)
    return dqn


def _create_and_train_madqn(env, device):
    madqn = MADQN(env, lr=1e-3, gamma=0.99, iteration=5,
                  epsilon=0.5, epsilon_decay=0.95, epsilon_min=0.1,
                  num_episodes=50, episode_length=5000, mini_batch_size=64,
                  update_freq=10, device=device)
    madqn, *_ = train_madqn(env, madqn)
    path = os.path.join(MODEL_DIR, "madqn_model.pth")
    madqn.save(path)
    return madqn


def _load_and_test_dqn(env, device, model_path, max_steps=200, verbose=True):
    dqn = DQN(env, agent_id=0, device=device)
    dqn.load(model_path)
    from rl_algorithms.test.test_dqn import run_with_agent
    return run_with_agent(env, dqn, max_steps=max_steps, debug=verbose)


def _load_and_test_madqn(env, device, model_path, max_steps=200, verbose=True):
    madqn = MADQN(env, device=device)
    madqn.load(model_path)
    from rl_algorithms.test.test_madqn import run_with_agent
    return run_with_agent(env, madqn, max_steps=max_steps, debug=verbose)


def main():
    parser = argparse.ArgumentParser(description="DQN/MADQN 训练与测试")
    parser.add_argument("--model", choices=["dqn", "madqn"], default="madqn", help="模型: dqn 或 madqn")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="模式: train 或 test")
    parser.add_argument("--model_path", type=str, default=None, help="测试时模型路径，默认 models/<model>_model.pth")
    parser.add_argument("--max_steps", type=int, default=200, help="测试时最大步数")
    parser.add_argument("--quiet", action="store_true", help="测试时减少输出")
    args = parser.parse_args()

    device = _get_device()
    env = _create_env()

    if args.mode == "train":
        print(f"训练 {args.model.upper()} ...")
        if args.model == "dqn":
            _create_and_train_dqn(env, device)
        else:
            _create_and_train_madqn(env, device)
        print("训练完成。")
        return

    # test
    path = args.model_path or os.path.join(MODEL_DIR, f"{args.model}_model.pth")
    if not os.path.isfile(path):
        print(f"未找到模型: {path}，请先训练或指定 --model_path")
        sys.exit(1)
    print(f"测试 {args.model.upper()}，加载: {path}")
    ok = _load_and_test_dqn(env, device, path, args.max_steps, not args.quiet) if args.model == "dqn" \
         else _load_and_test_madqn(env, device, path, args.max_steps, not args.quiet)
    print("测试通过。" if ok else "测试未在步数内完成。")


if __name__ == "__main__":
    main()
