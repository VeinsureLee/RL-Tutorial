import sys
import os
import numpy as np
import random
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.env import Env
from rl_algorithms.utils.agent import Agent


def run_with_agent(env, agents, max_steps=200, debug=False):
    """
    使用Agent策略运行环境，支持根据环境参数创建的多Agent独立行动
    :param env: 环境实例
    :param agents: Agent实例或Agent列表
    :param max_steps: 最大步数
    :return: 是否所有agent都到达目标
    """
    # 将单个Agent转换为列表，长度不足时复制引用以保持兼容
    if not isinstance(agents, (list, tuple)):
        agents = [agents for _ in range(env.num_agents)]
    elif len(agents) != env.num_agents:
        # 直接抛出异常提示配置错误，避免静默错位
        raise ValueError(f"传入的Agent数量({len(agents)})与环境配置的num_agents({env.num_agents})不一致")

    states, _ = env.reset()
    print(f"Agent数量: {env.num_agents}")
    print(f"初始状态: {states}")
    print(f"目标状态: {env.target_states}")
    print(f"动作空间: {env.action_space}")
    print(f"网格大小: {env.grid_rows}x{env.grid_cols}")
    print("-" * 50)
    
    action_names = {(0, 1): "下", (1, 0): "右", (0, -1): "上", (-1, 0): "左", (0, 0): "停留"}
    
    for step in range(max_steps):
        actions = []

        # 为每个Agent独立选择动作
        for agent_id in range(env.num_agents):
            state_i = states[agent_id] if isinstance(states, list) else states
            action = agents[agent_id].select_action(state_i, agent_id=agent_id, training=False)
            actions.append(action)
        
        # 执行动作
        next_states, rewards, dones, _ = env.step(actions)
        
        # 确保返回的是列表格式（兼容单agent情况）
        if not isinstance(next_states, list):
            next_states = [next_states]
        if not isinstance(rewards, list):
            rewards = [rewards]
        if not isinstance(dones, list):
            dones = [dones]
            
        # 打印信息
        if debug:
            action_str = ", ".join([f"Agent{i+1}:{action_names.get(a, str(a))}" for i, a in enumerate(actions)])
            state_str = ", ".join([f"Agent{i+1}:{s}" for i, s in enumerate(next_states)])
            reward_str = ", ".join([f"Agent{i+1}:{r:.1f}" for i, r in enumerate(rewards)])
            print(f"步数 {step+1}: 动作=[{action_str}], 状态=[{state_str}], 奖励=[{reward_str}]")
        
        # 更新状态
        states = next_states
        
        # 检查是否所有agent都完成
        if all(dones):
            print(f"\n所有Agent成功到达目标！总步数: {step+1}")
            return True
    
    print(f"\n达到最大步数 {max_steps}，未全部到达目标")
    return False


def main():
    """
    主函数：创建环境和Agent，运行Agent策略并渲染
    """
    print("=" * 50)
    print("使用Agent策略测试")
    print("=" * 50)
    
    # 创建环境
    env = Env()
    
    # 根据环境参数创建多个Agent实例（每个Agent独立决策）
    agents = [Agent(env) for _ in range(env.num_agents)]
    
    start_time = time.time()
    # 使用Agent运行环境
    success = run_with_agent(env, agents, max_steps=400000, debug=False)
    end_time = time.time()
    print(f"运行时间: {end_time - start_time}秒")
    
    print("\n" + "=" * 50)
    print("开始渲染环境")
    print("=" * 50)
    
    # 渲染静态图像
    print("\n渲染静态图像...")
    env.render(mode='human')
    
    # 渲染动画
    print("\n渲染动画...")
    try:
        env.render_animation(interval=1, save_path=None)
    except Exception as e:
        print(f"动画渲染出错: {e}")
        print("尝试使用静态渲染模式")
        env.render(mode='human')
    
    print("\n渲染完成！")


if __name__ == '__main__':
    main()

