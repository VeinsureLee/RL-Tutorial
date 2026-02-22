"""MADQN 预训练模型测试与双视图可视化。"""
import sys
import os
import time
import numpy as np
import random

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env.env import Env
from rl_algorithms.structure.madqn import MADQN
from rl_algorithms.test.visualize_dual import render_dual, render_animation_dual


def run_with_agent(env, agents, max_steps=200, debug=False):
    if not isinstance(agents, (list, tuple)):
        agents = [agents for _ in range(env.num_agents)]
    elif len(agents) != env.num_agents:
        raise ValueError(f"传入的Agent数量({len(agents)})与环境配置的num_agents({env.num_agents})不一致")

    states, _ = env.reset()
    if env.num_agents == 1:
        states = [states]
    print(f"Agent数量: {env.num_agents}")
    print(f"初始状态: {states}")
    print(f"目标状态: {env.target_states}")
    print("-" * 50)
    action_names = {(0, 1): "下", (1, 0): "右", (0, -1): "上", (-1, 0): "左", (0, 0): "停留"}

    # 传入单个 MADQN 时 agents 被展开为 [madqn, madqn, ...]，用 agents[0] 判断类型并调用 take_action
    agent = agents[0]
    for step in range(max_steps):
        if isinstance(agent, MADQN):
            action_indices = agent.take_action(states, training=False)
            actions = [env.action_space[action_idx] for action_idx in action_indices]
        else:
            actions = [agents[i].select_action(states[i] if isinstance(states, list) else states, agent_id=i, training=False)
                       for i in range(env.num_agents)]
        next_states, rewards, dones, _ = env.step(actions)
        if not isinstance(next_states, list):
            next_states, rewards, dones = [next_states], [rewards], [dones]
        if debug:
            action_str = ", ".join([f"Agent{i+1}:{action_names.get(a, str(a))}" for i, a in enumerate(actions)])
            state_str = ", ".join([f"Agent{i+1}:{s}" for i, s in enumerate(next_states)])
            print(f"步数 {step+1}: 动作=[{action_str}], 状态=[{state_str}]")
        states = next_states
        if all(dones):
            print(f"\n所有Agent成功到达目标！总步数: {step+1}")
            return True
    print(f"\n达到最大步数 {max_steps}，未全部到达目标")
    return False


def main():
    print("=" * 50)
    print("MADQN预训练模型测试与双视图可视化")
    print("=" * 50)
    env = Env()
    model_path = os.path.join(_ROOT, "models", "madqn_model.pth")
    madqn = MADQN(env, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9,
                  num_episodes=5, episode_length=35000, iteration=5, batch_size=64, mini_batch_size=64,
                  hidden_dim=128, update_freq=10)
    try:
        madqn.load(model_path)
        print(f"已加载模型: {model_path}")
    except FileNotFoundError:
        print(f"未找到预训练模型，请先运行 rl_algorithms.train 训练并保存到 {model_path}")
        return
    madqn.epsilon = madqn.epsilon_min
    print(f"\n测试预训练的MADQN模型（{env.num_agents}个机器人）")
    start_time = time.time()
    success = run_with_agent(env, madqn, max_steps=200, debug=False)
    print(f"运行时间: {time.time() - start_time}秒")

    print("\n开始渲染环境")
    png_dir = os.path.join(_ROOT, "results", "png")
    gif_dir = os.path.join(_ROOT, "results", "gif")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, "madqn_pretrained_test04.gif")
    last_frame_path = os.path.join(png_dir, "madqn_pretrained_test04_last_frame.png")
    # 先展示：双视图静态图 + 双视图动画
    render_dual(env, save_path=None)
    try:
        render_animation_dual(env, interval=1, save_gif_path=None, save_last_frame_path=None)
    except Exception as e:
        print(f"动画渲染出错: {e}")
        render_dual(env, save_path=None)
    # 后保存：双视图最后一帧 PNG + 双视图 GIF
    render_dual(env, save_path=last_frame_path, save_only=True)
    try:
        render_animation_dual(env, interval=200, save_gif_path=gif_path, save_last_frame_path=None, max_frames=100, save_only=True)
    except Exception as e:
        print(f"GIF 保存出错: {e}")
    print("\n渲染完成！")


if __name__ == "__main__":
    main()
