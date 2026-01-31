import sys
import os
import numpy as np
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.env import Env
from rl_algorithms.utils.agent import Agent
from rl_algorithms.dqn import DQN
from simulation.visualize_dual import render_dual, render_animation_dual


def run_with_agent(env, agent, max_steps=400000, training=False, verbose=True):
    """
    使用Agent策略运行环境
    :param env: 环境实例
    :param agent: Agent实例
    :param max_steps: 最大步数
    :param training: 是否处于训练模式
    :param verbose: 是否打印详细信息
    :return: 是否所有agent都到达目标
    """
    states, _ = env.reset()
    if verbose:
        print(f"Agent数量: {env.num_agents}")
        print(f"初始状态: {states}")
        print(f"目标状态: {env.target_states}")
        print(f"动作空间: {env.action_space}")
        print(f"网格大小: {env.grid_rows}x{env.grid_cols}")
        print("-" * 50)
    
    action_names = {(0, 1): "下", (1, 0): "右", (0, -1): "上", (-1, 0): "左", (0, 0): "停留"}
    
    # 对于DQN，确保agent_id已设置
    if isinstance(agent, DQN):
        if agent.agent_id is None:
            agent.agent_id = 0
        if env.num_agents > 1:
            print(f"警告: DQN是为单agent设计的，当前环境有{env.num_agents}个agent，将只使用agent_id={agent.agent_id}")
    
    for step in range(max_steps):
        # 使用agent选择动作
        if isinstance(agent, DQN):
            # DQN是为单agent设计的，只处理agent_id对应的agent
            agent_id = agent.agent_id if agent.agent_id is not None else 0
            state_i = states[agent_id] if isinstance(states, list) else states
            action_idx = agent.take_action(state_i, training=training)
            # 将索引转换为环境动作
            if isinstance(action_idx, (int, np.integer)) and 0 <= action_idx < env.num_actions:
                action = env.action_space[action_idx]
            else:
                action = action_idx
            
            # 对于多agent环境，其他agent保持不动（使用停留动作）
            if env.num_agents > 1:
                actions = [(0, 0) for _ in range(env.num_agents)]
                actions[agent_id] = action
            else:
                actions = [action]
        else:
            actions = agent.select_action(states, training=training)
            
            # 如果返回的是单个动作，转换为列表（兼容单agent情况）
            if not isinstance(actions, (list, tuple, np.ndarray)):
                actions = [actions]
            elif isinstance(actions, tuple) and len(actions) == 2:
                # 检查是否是动作tuple还是状态tuple
                if isinstance(actions[0], (int, float, np.integer, np.floating)):
                    actions = [actions]
            
            # 确保actions是列表格式
            if not isinstance(actions, list):
                actions = list(actions) if hasattr(actions, '__iter__') else [actions]
            
            # 确保actions数量正确
            if len(actions) != env.num_agents:
                if env.num_agents == 1:
                    actions = [actions[0] if len(actions) > 0 else random.choice(env.action_space)]
                else:
                    actions = [random.choice(env.action_space) for _ in range(env.num_agents)]
        
        # 执行动作
        next_states, rewards, dones, _ = env.step(actions)
        # 兼容单agent返回标量的情况
        if not isinstance(next_states, list):
            next_states = [next_states]
        if not isinstance(rewards, list):
            rewards = [rewards]
        if not isinstance(dones, list):
            dones = [dones]
        
        # run_with_agent 主要用于评估阶段，训练请使用专用训练流程
        
        # 打印信息
        if verbose:
            action_str = ", ".join([f"Agent{i+1}:{action_names.get(a, str(a))}" for i, a in enumerate(actions)])
            state_str = ", ".join([f"Agent{i+1}:{s}" for i, s in enumerate(next_states)])
            reward_str = ", ".join([f"Agent{i+1}:{r:.1f}" for i, r in enumerate(rewards)])
            print(f"步数 {step+1}: 动作=[{action_str}], 状态=[{state_str}], 奖励=[{reward_str}]")
        
        # 更新状态
        states = next_states
        
        # 检查是否所有agent都完成
        if all(dones):
            if verbose:
                print(f"\n所有Agent成功到达目标！总步数: {step+1}")
            return True
    
    if verbose:
        print(f"\n达到最大步数 {max_steps}，未全部到达目标")
    return False


def main():
    """
    主函数：创建环境和Agent，运行Agent策略并渲染
    """
    print("=" * 50)
    print("DQN预训练模型测试与可视化")
    print("=" * 50)
    
    # 创建环境
    env = Env()
    
    # 创建DQN Agent并加载预训练权重
    print("\n加载预训练的DQN模型...")
    model_path = os.path.join("models", "dqn_model_test.pth")
    fallback_model_path = os.path.join("models", "dqn_model_test.pth")
    dqn = DQN(env, agent_id=0, lr=0.001, gamma=0.99, iteration=10,
              epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
              num_episodes=50, episode_length=2000, batch_size=64, 
              mini_batch_size=32, hidden_dim=128, update_freq=50)

    try:
        dqn.load(model_path)
        print(f"已加载模型: {model_path}")
    except FileNotFoundError:
        try:
            dqn.load(fallback_model_path)
            model_path = fallback_model_path
            print(f"已加载模型: {fallback_model_path}")
        except FileNotFoundError:
            print(f"未找到预训练模型文件，请将文件放在 {model_path} 或 {fallback_model_path}")
            return
    
    # 使用训练好的策略（固定epsilon）
    dqn.epsilon = dqn.epsilon_min
    
    print("\n" + "=" * 50)
    print("测试预训练的DQN模型（单机器人）")
    print("=" * 50)
    
    success = run_with_agent(env, dqn, max_steps=2000, training=False, verbose=True)
    
    print("\n" + "=" * 50)
    print("开始渲染环境（热力图 + 普通图，图例在下方）")
    print("=" * 50)
    
    os.makedirs("results", exist_ok=True)
    gif_path = os.path.join("results/gif", "dqn_pretrained_test.gif")
    last_frame_path = os.path.join("results/png", "dqn_pretrained_test_last_frame.png")
    
    # 双视图静态图：热力图 + 普通图，地图上不显示 start/target，下方图例展示
    print("\n渲染双视图静态图（热力图 + 轨迹）...")
    try:
        render_dual(env, save_path=None)
    except Exception as e:
        print(f"双视图渲染出错: {e}")
    
    # 双视图动画：保存 GIF，并保存最后一帧
    print("\n渲染双视图动画并保存 GIF 与最后一帧...")
    try:
        render_animation_dual(
            env,
            interval=200,
            save_gif_path=gif_path,
            save_last_frame_path=last_frame_path,
            max_frames=100,
        )
        print(f"GIF 已保存: {gif_path}")
        print(f"最后一帧已保存: {last_frame_path}")
    except Exception as e:
        print(f"动画渲染出错: {e}")
        print("尝试使用静态双视图")
        render_dual(env, save_path=last_frame_path)
    
    print("\n渲染完成！")


if __name__ == '__main__':
    main()

