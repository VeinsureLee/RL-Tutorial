"""CLI 入口：训练 / 测试 RL 算法在室内多智能体环境中。

用法示例::
    python main.py --algo dqn --mode train
    python main.py --algo dqn --mode test --model_path experiments/<run>/model.pth
    python main.py --algo dqn --mode train --num_episodes 50 --lr 5e-5

参数优先级：命令行 > config/config.yml > 代码默认值。
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from algorithms import build_algorithm
from core.plot import plot_training
from core.tester import test
from core.trainer import train
from envs import IndoorEnv
from utils.config import load_config, merge_overrides
from utils.logger import get_logger
from utils.paths import experiments_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, help="Algorithm name from registry")
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--map", default=None, help="Map file name (without .yml)")
    p.add_argument("--model_path", default=None, help="Checkpoint path for test mode")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tag", default="", help="Appended to run id")
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_run_dir(algo: str, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    d = experiments_dir() / f"{stamp}_{algo}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    args = _parse_args()
    cfg = load_config()

    overrides: dict = {"algorithm": {"name": args.algo}}
    if args.map:
        overrides["env"] = {"map_file": args.map}
    if args.num_episodes is not None:
        overrides["algorithm"]["num_episodes"] = args.num_episodes
    if args.lr is not None:
        overrides["algorithm"]["lr"] = args.lr
    cfg = merge_overrides(cfg, overrides)

    _seed_everything(cfg["seed"])
    run_dir = _make_run_dir(args.algo, args.tag)
    logger = get_logger("main", log_file=run_dir / "run.log")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Algorithm: {args.algo}, mode: {args.mode}")

    env = IndoorEnv(cfg)
    algo = build_algorithm(args.algo, env, cfg)

    if args.mode == "train":
        result = train(algo, env, cfg, run_dir)
        plot_training(result.history, run_dir / "figs")
        logger.info(f"Training complete. Model: {result.model_path}")
    else:
        if args.model_path:
            algo.load(args.model_path)
        result = test(algo, env, cfg, run_dir / "test", max_steps=args.max_steps)
        logger.info(
            f"Test: success={result.success} steps={result.steps} "
            f"reward={result.total_reward:.2f}"
        )


if __name__ == "__main__":
    main()
