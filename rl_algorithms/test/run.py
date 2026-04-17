"""
统一测试入口：支持通过函数参数加载模型并测试 DQN / MADQN。

用法:
    from rl_algorithms.test.run import test
    test()
    test(algo="madqn", model_path="models/madqn_model.pth", max_steps=300)
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from env.env import MultiRobotEnv
from config.yml_config import get_env_config, get_rl_config
from rl_algorithms.structure.dqn import DQN
from rl_algorithms.structure.madqn import MADQN


def _run_episode(env, model, algo, max_steps):
    """运行一个测试 episode，返回 (success, steps, total_reward, ber_history, frames_data)。"""
    states = env.reset()
    total_reward = 0.0
    ber_history = []
    frames_data = []

    # 记录初始帧
    frames_data.append((
        env.positions.copy(),
        env.done_flags.copy(),
        [list(t) for t in env.trajectories],
    ))

    for step in range(max_steps):
        if algo == "madqn":
            actions = model.take_action(states, training=False)
        else:
            state_i = states[model.agent_id]
            action = model.take_action(state_i, training=False)
            actions = [np.random.randint(env.n_actions) for _ in range(env.num_agents)]
            actions[model.agent_id] = action

        next_states, rewards, dones, info = env.step(actions)
        total_reward += sum(rewards)
        ber_history.append(float(info["ber"].mean()))

        # 记录帧
        frames_data.append((
            env.positions.copy(),
            env.done_flags.copy(),
            [list(t) for t in env.trajectories],
        ))

        states = next_states
        if env.all_done:
            return True, step + 1, total_reward, ber_history, frames_data

    return False, max_steps, total_reward, ber_history, frames_data


def test(algo=None, env=None, model=None, model_path=None,
         max_steps=None, save_results=True, **kwargs):
    """统一测试接口。"""
    cfg = get_rl_config(algo=algo, **kwargs)
    algo = cfg["algo"]
    if max_steps is None:
        max_steps = cfg["test_max_steps"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env is None:
        env_config = get_env_config()
        env = MultiRobotEnv(env_config)

    if model is None:
        common = dict(
            lr=cfg["lr"], gamma=cfg["gamma"],
            epsilon=cfg["epsilon_min"], epsilon_min=cfg["epsilon_min"],
            epsilon_decay=cfg["epsilon_decay"],
            hidden_dim=cfg["hidden_dim"], update_freq=cfg["update_freq"],
            replay_buffer_size=cfg["replay_buffer_size"], device=device,
        )
        if model_path is None:
            model_dir = os.path.join(_ROOT, cfg["model_dir"])
            new_name = os.path.join(model_dir, f"{algo}_{env.num_agents}agents_model.pth")
            old_name = os.path.join(model_dir, f"{algo}_model.pth")
            model_path = new_name if os.path.isfile(new_name) else old_name

        if algo == "dqn":
            model = DQN(env, agent_id=0, **common)
        else:
            model = MADQN(env, **common)

        model.load(model_path)
        model.epsilon = model.epsilon_min
        print(f"已加载模型: {model_path}")

    print(f"\n{'='*50}")
    print(f"测试 {algo.upper()} ({env.num_agents} agents, max_steps={max_steps})")
    print(f"{'='*50}")

    start_time = time.time()
    success, steps, total_reward, ber_history, frames_data = _run_episode(env, model, algo, max_steps)
    elapsed = time.time() - start_time

    if success:
        print(f"\n所有 Agent 成功到达目标！步数: {steps}")
    else:
        print(f"\n达到最大步数 {max_steps}，未全部到达")
    print(f"总奖励: {total_reward:.2f}，耗时: {elapsed:.2f}s")
    if ber_history:
        print(f"平均 BER: {np.mean(ber_history):.4e}")

    result = dict(
        success=success, steps=steps, total_reward=total_reward,
        ber_history=ber_history, model=model, algo=algo,
    )

    if save_results:
        import matplotlib.pyplot as plt

        gif_dir = os.path.join(_ROOT, "results", "gif")
        png_dir = os.path.join(_ROOT, "results", "png")
        os.makedirs(gif_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        prefix = f"{algo}_{env.num_agents}agents"

        # --- 导航 GIF ---
        nav_gif = os.path.join(gif_dir, f"{prefix}_nav.gif")
        try:
            env.save_nav_gif(nav_gif, frames_data, fps=5)
            print(f"导航 GIF 已保存: {nav_gif}")
            result["nav_gif"] = nav_gif
        except Exception as e:
            print(f"导航 GIF 保存出错: {e}")

        # --- 信号 GIF ---
        sig_gif = os.path.join(gif_dir, f"{prefix}_signal.gif")
        try:
            env.save_signal_gif(sig_gif, frames_data, fps=5)
            print(f"信号 GIF 已保存: {sig_gif}")
            result["signal_gif"] = sig_gif
        except Exception as e:
            print(f"信号 GIF 保存出错: {e}")

        # --- 导航地图 PNG（白色背景）---
        try:
            nav_path = os.path.join(png_dir, f"{prefix}_nav.png")
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            env.render_nav_frame(ax)
            fig.savefig(nav_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"导航图已保存: {nav_path}")
            result["nav_path"] = nav_path
        except Exception as e:
            print(f"导航图保存出错: {e}")

        # --- 通信质量地图 PNG（热力图背景）---
        try:
            sig_path = os.path.join(png_dir, f"{prefix}_signal.png")
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            env.render_signal_frame(ax)
            fig.savefig(sig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"信号图已保存: {sig_path}")
            result["signal_path"] = sig_path
        except Exception as e:
            print(f"信号图保存出错: {e}")

    return result


def main():
    test()
