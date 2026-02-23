"""
训练调试入口：运行 DQN 或 MADQN，训练后保存模型并绘图。
供 rl_algorithms.train.main 调用或 python -m rl_algorithms.train 使用。
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

def main():
    """调试入口：可选运行 DQN 或 MADQN，训练后保存模型并绘图到 rl_algorithms/figs。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env()
    algo = "madqn"

    if algo == "dqn":
        dqn = DQN(env, agent_id=0, lr=1e-3, gamma=0.99, iteration=5,
                  epsilon=0.8, epsilon_decay=0.9, epsilon_min=0.1,
                  num_episodes=50, episode_length=35000, mini_batch_size=64,
                  update_freq=10, device=device)
        dqn, return_list = train_dqn(env, dqn)
        dqn.save(os.path.join(_ROOT, "models", "dqn_model.pth"))
        path = plot_dqn(return_list, save_prefix="dqn")
        print(f"训练曲线已保存: {path}")
    else:
        madqn = MADQN(env, lr=1e-3, gamma=0.99, iteration=5,
                      epsilon=0.5, epsilon_decay=0.95, epsilon_min=0.1,
                      num_episodes=50, episode_length=5000, mini_batch_size=64,
                      update_freq=10, device=device)
        madqn, return_list, agent_return_lists, ber_list, agent_ber_lists = train_madqn(env, madqn)
        madqn.save(os.path.join(_ROOT, "models", "madqn_model.pth"))
        paths = plot_madqn(return_list, agent_return_lists, ber_list, agent_ber_lists, save_prefix="madqn")
        print(f"训练曲线已保存至: {FIG_DIR}")
        for p in paths:
            print(f"  - {p}")
