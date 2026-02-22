"""DQN 预训练模型测试与双视图可视化。"""
import sys
import os
import time
import numpy as np
import random

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env.env import Env
from rl_algorithms.utils.agent import Agent
from rl_algorithms.structure.dqn import DQN
from rl_algorithms.test.visualize_dual import render_dual, render_animation_dual


def run_with_agent(env, agents, max_steps=200, debug=False):
    if not isinstance(agents, (list, tuple)):
        agents = [agents for _ in range(env.num_agents)]
    elif len(agents) != env.num_agents:
        raise ValueError(f"传入的Agent数量({len(agents)})与环境配置的num_agents({env.num_agents})不一致")

    states, _ = env.reset()
    print(f"Agent数量: {env.num_agents}")
    print(f"初始状态: {states}")
    print(f"目标状态: {env.target_states}")
    print("-" * 50)
    action_names = {(0, 1): "下", (1, 0): "右", (0, -1): "上", (-1, 0): "左", (0, 0): "停留"}

    agent = agents[0] if env.num_agents == 1 else agents
    if hasattr(agent, "agent_id") and agent.agent_id is None:
        agent.agent_id = 0
    if hasattr(agent, "agent_id") and env.num_agents > 1 and debug:
        print(f"警告: DQN 为单 agent 设计，当前环境有 {env.num_agents} 个 agent，将只使用 agent_id={agent.agent_id}")

    for step in range(max_steps):
        if hasattr(agent, "take_action") and hasattr(agent, "agent_id"):
            state_i = states[agent.agent_id] if isinstance(states, list) else states
            action_idx = agent.take_action(state_i, training=False)
            action = env.action_space[action_idx] if isinstance(action_idx, (int, np.integer)) and 0 <= action_idx < env.num_actions else action_idx
            actions = [(0, 0) for _ in range(env.num_agents)] if env.num_agents > 1 else [action]
            if env.num_agents > 1:
                actions[agent.agent_id] = action
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
    print("DQN预训练模型测试与双视图可视化")
    print("=" * 50)
    env = Env()
    model_path = os.path.join(_ROOT, "models", "dqn_model.pth")
    dqn = DQN(env, agent_id=0, lr=0.001, gamma=0.99, iteration=10, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
              num_episodes=50, episode_length=2000, batch_size=64, mini_batch_size=32, hidden_dim=128, update_freq=50)
    try:
        dqn.load(model_path)
        print(f"已加载模型: {model_path}")
    except FileNotFoundError:
        print(f"未找到预训练模型，请先运行 rl_algorithms.train 训练并保存到 {model_path}")
        return
    dqn.epsilon = dqn.epsilon_min
    print("\n测试预训练的DQN模型（单机器人）")
    start_time = time.time()
    success = run_with_agent(env, dqn, max_steps=200, debug=False)
    print(f"运行时间: {time.time() - start_time}秒")

    print("\n开始渲染环境")
    png_dir = os.path.join(_ROOT, "results", "png")
    gif_dir = os.path.join(_ROOT, "results", "gif")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, "dqn_pretrained_test.gif")
    last_frame_path = os.path.join(png_dir, "dqn_pretrained_test_last_frame.png")
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
