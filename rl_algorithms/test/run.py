"""
统一测试入口：支持通过函数参数加载模型并测试 DQN / MADQN。

用法:
    # 1. 使用 yml 默认参数
    from rl_algorithms.test.run import test
    test()

    # 2. 传入自定义参数
    test(algo="madqn", model_path="models/madqn_model.pth", max_steps=300)

    # 3. 测试后不渲染动画
    test(render=False)
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from env.env import Env
from rl_algorithms.structure.dqn import DQN
from rl_algorithms.structure.madqn import MADQN
from rl_algorithms.test.visualize_dual import render_dual, render_animation_dual
from config.yml_config import get_rl_config


def _run_episode(env, model, algo, max_steps, debug=False):
    """运行一个测试 episode，返回 (success, steps, total_reward, ber_history)。"""
    states, _ = env.reset()  # 始终返回 list of tuples

    action_names = {(0, 1): "下", (1, 0): "右", (0, -1): "上", (-1, 0): "左", (0, 0): "停留"}
    total_reward = 0.0
    ber_history = []

    for step in range(max_steps):
        if algo == "madqn":
            action_indices = model.take_action(states, training=False)
            actions = [env.action_space[idx] for idx in action_indices]
        else:
            state_i = states[model.agent_id] if isinstance(states, list) else states
            action_idx = model.take_action(state_i, training=False)
            actions = [(0, 0)] * env.num_agents
            actions[model.agent_id] = env.action_space[action_idx]

        step_result = env.step(actions)
        if env.num_agents == 1:
            ns, r, d, info = step_result
            next_states, rewards, dones = [ns], [r], [d]
        else:
            next_states, rewards, dones, info = step_result

        total_reward += sum(rewards)
        ber = info.get("ber_per_agent", info.get("ber", []))
        if ber:
            ber_history.append(np.mean(ber))

        if debug:
            a_str = ", ".join([f"A{i+1}:{action_names.get(a, str(a))}" for i, a in enumerate(actions)])
            print(f"  Step {step+1}: actions=[{a_str}]")

        states = next_states
        if all(dones):
            return True, step + 1, total_reward, ber_history

    return False, max_steps, total_reward, ber_history


def test(algo=None, env=None, model=None, model_path=None,
         max_steps=None, render=True, save_results=True, debug=False, **kwargs):
    """
    统一测试接口。所有参数可选，未传入的从 config/base/rl.yml 读取。

    :param algo:         算法 "dqn" 或 "madqn"
    :param env:          环境实例，None 则自动创建
    :param model:        已加载的模型实例，None 则根据 model_path 加载
    :param model_path:   模型文件路径，None 则使用默认路径
    :param max_steps:    测试最大步数
    :param render:       是否渲染可视化
    :param save_results: 是否保存 PNG/GIF
    :param debug:        是否打印每步信息
    :param kwargs:       覆盖 rl.yml 中的任意参数（用于构建模型）
    :return: dict 包含测试结果:
        success, steps, total_reward, ber_history, model
    """
    cfg = get_rl_config(algo=algo, **kwargs)
    algo = cfg["algo"]
    if max_steps is None:
        max_steps = cfg["test_max_steps"]

    if env is None:
        env = Env()

    # ---- 加载/构建模型 ----
    if model is None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        common = dict(
            lr=cfg["lr"], gamma=cfg["gamma"],
            epsilon=cfg["epsilon_min"], epsilon_min=cfg["epsilon_min"],
            epsilon_decay=cfg["epsilon_decay"],
            num_episodes=cfg["num_episodes"], episode_length=cfg["episode_length"],
            iteration=cfg["iteration"], batch_size=cfg["batch_size"],
            mini_batch_size=cfg["mini_batch_size"], hidden_dim=cfg["hidden_dim"],
            update_freq=cfg["update_freq"], device=device,
        )
        if model_path is None:
            model_dir = os.path.join(_ROOT, cfg["model_dir"])
            # 优先找 algo_Nagents_model.pth，兼容旧命名 algo_model.pth
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

    # ---- 运行测试 ----
    print(f"\n{'='*50}")
    print(f"测试 {algo.upper()} ({env.num_agents} agents, max_steps={max_steps})")
    print(f"{'='*50}")

    start_time = time.time()
    success, steps, total_reward, ber_history = _run_episode(env, model, algo, max_steps, debug)
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

    # ---- 渲染可视化 ----
    if render or save_results:
        png_dir = os.path.join(_ROOT, "results", "png")
        gif_dir = os.path.join(_ROOT, "results", "gif")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)

        prefix = f"{algo}_{env.num_agents}agents"
        png_path = os.path.join(png_dir, f"{prefix}_test.png")
        gif_path = os.path.join(gif_dir, f"{prefix}_test.gif")
        last_frame_path = os.path.join(png_dir, f"{prefix}_test_last_frame.png")

        if render:
            render_dual(env, save_path=None)
            try:
                render_animation_dual(env, interval=1, save_gif_path=None, save_last_frame_path=None)
            except Exception as e:
                print(f"动画渲染出错: {e}")

        if save_results:
            render_dual(env, save_path=png_path, save_only=True)
            try:
                render_animation_dual(env, interval=200, save_gif_path=gif_path,
                                      save_last_frame_path=last_frame_path,
                                      max_frames=100, save_only=True)
            except Exception as e:
                print(f"GIF 保存出错: {e}")
            result["png_path"] = png_path
            result["gif_path"] = gif_path
            result["last_frame_path"] = last_frame_path

    return result


def main():
    """命令行入口，等价于 test()。"""
    test()
