import sys
import os
import numpy as np
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.env import Env


def random_policy(env, max_steps=200):
    """
    随机策略：在每个步骤中为每个agent随机选择一个动作
    :param env: 环境实例
    :param max_steps: 最大步数
    :return: 是否所有agent都到达目标
    """
    states, _ = env.reset()
    print(f"Agent数量: {env.num_agents}")
    print(f"初始状态: {states}")
    print(f"目标状态: {env.target_states}")
    print(f"动作空间: {env.action_space}")
    print(f"网格大小: {env.grid_rows}x{env.grid_cols}")
    print("-" * 50)
    
    action_names = {(0, 1): "下", (1, 0): "右", (0, -1): "上", (-1, 0): "左", (0, 0): "停留"}
    
    for step in range(max_steps):
        # 为每个agent随机选择一个动作
        actions = [random.choice(env.action_space) for _ in range(env.num_agents)]
        
        # 执行动作
        next_states, rewards, dones, _ = env.step(actions)
        
        # 打印信息
        action_str = ", ".join([f"Agent{i+1}:{action_names.get(a, str(a))}" for i, a in enumerate(actions)])
        state_str = ", ".join([f"Agent{i+1}:{s}" for i, s in enumerate(next_states)])
        reward_str = ", ".join([f"Agent{i+1}:{r:.1f}" for i, r in enumerate(rewards)])
        print(f"步数 {step+1}: 动作=[{action_str}], 状态=[{state_str}], 奖励=[{reward_str}]")
        
        # 检查是否所有agent都完成
        if all(dones):
            print(f"\n所有Agent成功到达目标！总步数: {step+1}")
            return True
    
    print(f"\n达到最大步数 {max_steps}，未全部到达目标")
    return False


def main():
    """
    主函数：创建环境，运行随机策略并渲染
    """
    print("=" * 50)
    print("随机策略测试")
    print("=" * 50)
    
    # 创建环境
    env = Env()
    
    # 运行随机策略
    success = random_policy(env, max_steps=10000)
    
    print("\n" + "=" * 50)
    print("开始渲染环境")
    print("=" * 50)
    
    # 渲染静态图像
    print("\n渲染静态图像...")
    env.render(mode='human')
    
    # 渲染动画
    print("\n渲染动画...")
    try:
        env.render_animation(interval=10, save_path=None)
    except Exception as e:
        print(f"动画渲染出错: {e}")
        print("尝试使用静态渲染模式")
        env.render(mode='human')
    
    print("\n渲染完成！")


if __name__ == '__main__':
    main()

