"""
项目入口：在此统一完成参数解析、环境 & 模型构建、模式分派（train / test）。

用法示例:
    python main.py --model madqn --mode train
    python main.py --model dqn   --mode train --num_episodes 200 --lr 5e-5
    python main.py --model madqn --mode test  --model_path models/madqn_4agents_model.pth
    python main.py --model dqn   --mode test  --max_steps 500

参数优先级: 命令行 > config/base/rl.yml > 代码内默认。
"""
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from env.env import MultiRobotEnv
from config.yml_config import get_env_config, get_rl_config
from rl_algorithms import DQN, MADQN, train, test, plot_training
from utils.logger_handler import get_logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Robot DRL Navigation (DQN / MADQN)")
    p.add_argument("--model", choices=["dqn", "madqn"], default="madqn", help="algorithm")
    p.add_argument("--mode", choices=["train", "test"], default="train", help="mode")

    # 训练超参（命令行覆盖 rl.yml）
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--epsilon_min", type=float, default=None)
    p.add_argument("--epsilon_decay", type=float, default=None)
    p.add_argument("--num_iterations", type=int, default=None)
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--episode_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--update_freq", type=int, default=None)
    p.add_argument("--replay_buffer_size", type=int, default=None)
    p.add_argument("--train_interval", type=int, default=None,
                   help="gradient update interval (env steps); None -> rl.yml value (3bede13 = 1)")

    # 测试参数
    p.add_argument("--model_path", type=str, default=None, help="path to .pth for test")
    p.add_argument("--max_steps", type=int, default=None, help="max steps during test")

    # 通用
    p.add_argument("--agent_id", type=int, default=0, help="DQN only: which agent to train")
    p.add_argument("--no_save_model", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def _resolve_cfg(args: argparse.Namespace) -> dict:
    """命令行 > yml 合并：非 None 的命令行字段覆盖 yml 默认值。"""
    overrides = {k: v for k, v in vars(args).items() if v is not None and k in {
        "lr", "gamma", "epsilon", "epsilon_min", "epsilon_decay",
        "num_iterations", "num_episodes", "episode_length",
        "batch_size", "hidden_dim", "update_freq", "replay_buffer_size",
        "train_interval",
    }}
    cfg = get_rl_config(algo=args.model, **overrides)
    return cfg


def _build_model(algo: str, env, cfg: dict, device: torch.device, agent_id: int = 0):
    """根据算法构建模型实例。"""
    common = dict(
        lr=cfg["lr"], gamma=cfg["gamma"],
        epsilon=cfg["epsilon"], epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        hidden_dim=cfg["hidden_dim"], update_freq=cfg["update_freq"],
        replay_buffer_size=cfg["replay_buffer_size"], device=device,
    )
    if algo == "dqn":
        return DQN(env, agent_id=agent_id, **common)
    return MADQN(env, **common)


def _run_train(args, cfg: dict, env, device: torch.device, prefix: str, model_dir: str) -> dict:
    model = _build_model(args.model, env, cfg, device, agent_id=args.agent_id)
    logger = get_logger(args.model)

    history = train(
        env=env, model=model,
        num_iterations=cfg["num_iterations"],
        num_episodes=cfg["num_episodes"],
        episode_length=cfg["episode_length"],
        batch_size=cfg["batch_size"],
        train_interval=cfg["train_interval"],
        logger=logger,
    )

    out = {"history": history}
    if not args.no_save_model:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{prefix}_model.pth")
        model.save(path)
        out["model_path"] = path
        print(f"saved model: {path}")

    if not args.no_plot:
        fig_dir = os.path.join(_ROOT, "results", "figs")
        paths = plot_training(history, fig_dir=fig_dir, prefix=prefix, algo_label=args.model.upper())
        out["fig_paths"] = paths
        for p in paths:
            print(f"saved figure: {p}")
    return out


def _run_test(args, cfg: dict, env, device: torch.device, prefix: str, model_dir: str) -> dict:
    model = _build_model(args.model, env, cfg, device, agent_id=args.agent_id)

    # 解析模型路径（新命名 -> 旧命名兜底）
    model_path = args.model_path
    if model_path is None:
        new_path = os.path.join(model_dir, f"{prefix}_model.pth")
        old_path = os.path.join(model_dir, f"{args.model}_model.pth")
        model_path = new_path if os.path.isfile(new_path) else old_path
    model.load(model_path)
    model.epsilon = model.epsilon_min
    print(f"loaded model: {model_path}")

    max_steps = args.max_steps if args.max_steps is not None else cfg["test_max_steps"]
    save_dir = os.path.join(_ROOT, "results")
    return test(env, model, max_steps=max_steps, save_dir=save_dir, prefix=prefix)


def main():
    args = _parse_args()
    cfg = _resolve_cfg(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MultiRobotEnv(get_env_config())

    prefix = f"{args.model}_{env.num_agents}agents"
    model_dir = os.path.join(_ROOT, cfg["model_dir"])

    if args.mode == "train":
        _run_train(args, cfg, env, device, prefix, model_dir)
    else:
        _run_test(args, cfg, env, device, prefix, model_dir)


if __name__ == "__main__":
    main()
