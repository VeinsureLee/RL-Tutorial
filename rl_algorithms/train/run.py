"""
统一训练入口：支持通过函数参数或 config/base/rl.yml 配置 DQN / MADQN 训练。

用法:
    # 1. 使用 yml 默认参数
    from rl_algorithms.train.run import train
    train()

    # 2. 传入自定义参数（覆盖 yml 默认值）
    train(algo="madqn", lr=1e-4, num_episodes=100, episode_length=3000)

    # 3. 命令行入口（使用 yml 默认参数）
    python -m rl_algorithms.train
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from env.env import Env
from rl_algorithms.structure.dqn import DQN
from rl_algorithms.structure.madqn import MADQN
from rl_algorithms.plot.plot import plot_dqn, plot_madqn, FIG_DIR
from rl_algorithms.train.train_dqn import train_dqn
from rl_algorithms.train.train_madqn import train_madqn
from config.yml_config import get_rl_config


def train(algo=None, env=None, device=None, save_model=True, plot=True, **kwargs):
    """
    统一训练接口。所有参数可选，未传入的从 config/base/rl.yml 读取。

    :param algo:           算法 "dqn" 或 "madqn"
    :param env:            环境实例，None 则自动创建
    :param device:         torch.device，None 则自动选择
    :param save_model:     训练后是否保存模型
    :param plot:           训练后是否绘图
    :param kwargs:         覆盖 rl.yml 中的任意超参数，支持:
        lr, gamma, epsilon, epsilon_min, epsilon_decay,
        iteration, num_episodes, episode_length,
        batch_size, mini_batch_size, hidden_dim, update_freq,
        model_dir
    :return: dict 包含训练结果:
        algo, model, return_list, agent_return_lists (madqn),
        ber_list (madqn), agent_ber_lists (madqn), model_path, fig_paths
    """
    # ---- 合并参数：kwargs > rl.yml > 默认值 ----
    cfg = get_rl_config(algo=algo, **kwargs)
    algo = cfg["algo"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if env is None:
        env = Env()

    model_dir = os.path.join(_ROOT, cfg["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    # ---- 公共超参数 ----
    common = dict(
        lr=cfg["lr"], gamma=cfg["gamma"],
        epsilon=cfg["epsilon"], epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        num_episodes=cfg["num_episodes"], episode_length=cfg["episode_length"],
        iteration=cfg["iteration"], batch_size=cfg["batch_size"],
        mini_batch_size=cfg["mini_batch_size"], hidden_dim=cfg["hidden_dim"],
        update_freq=cfg["update_freq"], device=device,
    )

    # 命名前缀：algo_Nagents，如 madqn_8agents
    num_agents = env.num_agents
    prefix = f"{algo}_{num_agents}agents"

    result = {"algo": algo, "config": cfg, "prefix": prefix}

    if algo == "dqn":
        model = DQN(env, agent_id=0, **common)
        model, return_list = train_dqn(env, model)
        result.update(model=model, return_list=return_list)

        if save_model:
            path = os.path.join(model_dir, f"{prefix}_model.pth")
            model.save(path)
            result["model_path"] = path

        if plot:
            fig = plot_dqn(return_list, save_prefix=prefix)
            result["fig_paths"] = [fig]
            print(f"训练曲线已保存: {fig}")

    elif algo == "madqn":
        model = MADQN(env, **common)
        model, return_list, agent_return_lists, ber_list, agent_ber_lists = train_madqn(env, model)
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
            print(f"训练曲线已保存至: {FIG_DIR}")
    else:
        raise ValueError(f"不支持的算法: {algo}，请使用 'dqn' 或 'madqn'")

    return result


# 保持向后兼容
def main():
    """命令行入口，等价于 train()。"""
    train()
