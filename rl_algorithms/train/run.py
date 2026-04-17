"""
统一训练入口：支持通过函数参数或 config/base/rl.yml 配置 DQN / MADQN 训练。

用法:
    from rl_algorithms.train.run import train
    train()
    train(algo="madqn", lr=1e-4, num_episodes=100)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from env.env import MultiRobotEnv
from config.yml_config import get_env_config, get_rl_config
from rl_algorithms.structure.dqn import DQN
from rl_algorithms.structure.madqn import MADQN
from rl_algorithms.plot.plot import plot_dqn, plot_madqn, FIG_DIR
from rl_algorithms.train.train_dqn import train_dqn
from rl_algorithms.train.train_madqn import train_madqn


def train(algo=None, env=None, device=None, save_model=True, plot=True, **kwargs):
    """
    统一训练接口。所有参数可选，未传入的从 config/base/rl.yml 读取。
    """
    cfg = get_rl_config(algo=algo, **kwargs)
    algo = cfg["algo"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if env is None:
        env_config = get_env_config()
        env = MultiRobotEnv(env_config)

    model_dir = os.path.join(_ROOT, cfg["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    num_agents = env.num_agents
    prefix = f"{algo}_{num_agents}agents"
    result = {"algo": algo, "config": cfg, "prefix": prefix}

    common = dict(
        lr=cfg["lr"], gamma=cfg["gamma"],
        epsilon=cfg["epsilon"], epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        hidden_dim=cfg["hidden_dim"], update_freq=cfg["update_freq"],
        replay_buffer_size=cfg["replay_buffer_size"], device=device,
    )
    train_kwargs = dict(
        num_iterations=cfg["num_iterations"],
        num_episodes=cfg["num_episodes"],
        episode_length=cfg["episode_length"],
        batch_size=cfg["batch_size"],
    )

    if algo == "dqn":
        model = DQN(env, agent_id=0, **common)
        model, return_list = train_dqn(env, model, **train_kwargs)
        result.update(model=model, return_list=return_list)

        if save_model:
            path = os.path.join(model_dir, f"{prefix}_model.pth")
            model.save(path)
            result["model_path"] = path

        if plot:
            fig = plot_dqn(return_list, save_prefix=prefix)
            result["fig_paths"] = [fig]

    elif algo == "madqn":
        model = MADQN(env, **common)
        model, return_list, agent_return_lists, ber_list, agent_ber_lists = \
            train_madqn(env, model, **train_kwargs)
        result.update(
            model=model, return_list=return_list,
            agent_return_lists=agent_return_lists,
            ber_list=ber_list, agent_ber_lists=agent_ber_lists,
        )

        if save_model:
            path = os.path.join(model_dir, f"{prefix}_model.pth")
            model.save(path)
            result["model_path"] = path

        if plot:
            figs = plot_madqn(return_list, agent_return_lists, ber_list, agent_ber_lists, save_prefix=prefix)
            result["fig_paths"] = figs

    else:
        raise ValueError(f"不支持的算法: {algo}，请使用 'dqn' 或 'madqn'")

    return result


def main():
    train()
