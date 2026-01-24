import sys
import os
import numpy as np
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.env import Env
from rl_algorithms.madqn import MADQN, train_madqn


def run_with_agent(env, agent, max_steps=400000, training=False, verbose=True):
    """
    使用MADQN Agent策略运行环境
    :param env: 环境实例
    :param agent: MADQN实例
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
    
    for step in range(max_steps):
        # 使用MADQN选择动作
        if isinstance(agent, MADQN):
            # 确保states是列表
            if not isinstance(states, list):
                states = [states]
            
            # 获取所有agent的id
            agent_ids = list(range(env.num_agents))
            
            # 使用MADQN的take_action方法（支持多agent）
            action_indices = agent.take_action(states, agent_ids, training=training)
            
            # 确保action_indices是列表
            if not isinstance(action_indices, list):
                action_indices = [action_indices]
            
            # 将索引转换为环境动作
            actions = []
            for action_idx in action_indices:
                if isinstance(action_idx, (int, np.integer)) and 0 <= action_idx < env.num_actions:
                    actions.append(env.action_space[action_idx])
                else:
                    actions.append(action_idx)
        else:
            # 兼容其他类型的agent
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


def train_madqn_model():
    """
    训练MADQN模型
    """
    print("=" * 50)
    print("MADQN模型训练")
    print("=" * 50)
    
    # 创建环境
    env = Env()
    
    # 创建MADQN Agent
    print("\n初始化MADQN模型...")
    madqn = MADQN(
        env,
        lr=0.001,
        gamma=0.99,
        iteration=5,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.95,
        num_episodes=50,
        episode_length=4000,
        batch_size=64,
        mini_batch_size=32,
        hidden_dim=128,
        update_freq=10
    )
    
    # 训练模型
    print("\n开始训练...")
    train_madqn(madqn)
    
    # 保存模型
    print("\n保存模型...")
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "madqn_model.pth")
    madqn.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    return madqn


def test_madqn_model():
    """
    测试预训练的MADQN模型
    """
    print("=" * 50)
    print("MADQN预训练模型测试与可视化")
    print("=" * 50)
    
    # 创建环境
    env = Env()
    
    # 创建MADQN Agent并加载预训练权重
    print("\n加载预训练的MADQN模型...")
    model_path = os.path.join("models", "models/madqn_model.pth")
    fallback_model_path = os.path.join("models", "models/madqn_model")
    
    madqn = MADQN(
        env,
        lr=0.001,
        gamma=0.99,
        iteration=5,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.95,
        num_episodes=50,
        episode_length=4000,
        batch_size=64,
        mini_batch_size=32,
        hidden_dim=128,
        update_freq=10
    )

    try:
        madqn.load(model_path)
        print(f"已加载模型: {model_path}")
    except FileNotFoundError:
        try:
            madqn.load(fallback_model_path)
            model_path = fallback_model_path
            print(f"已加载模型: {fallback_model_path}")
        except FileNotFoundError:
            print(f"未找到预训练模型文件，请将文件放在 {model_path} 或 {fallback_model_path}")
            print("提示：可以运行 train_madqn_model() 来训练模型")
            return None
    
    # 使用训练好的策略（设置epsilon为最小值）
    for agent_id in range(madqn.num_agents):
        madqn.epsilons[agent_id] = madqn.epsilon_min
    
    print("\n" + "=" * 50)
    print(f"测试预训练的MADQN模型（{env.num_agents}个机器人）")
    print("=" * 50)
    
    success = run_with_agent(env, madqn, max_steps=2000, training=False, verbose=True)
    
    print("\n" + "=" * 50)
    print("开始渲染环境")
    print("=" * 50)
    
    # 渲染静态图像
    print("\n渲染静态图像...")
    env.render(mode='human')
    
    # 渲染动画并保存
    print("\n渲染动画...")
    os.makedirs("results", exist_ok=True)
    gif_path = os.path.join("results", "madqn_pretrained_test.gif")
    try:
        env.render_animation(interval=1, save_path=gif_path)
        print(f"动画已保存到: {gif_path}")
    except Exception as e:
        print(f"动画渲染出错: {e}")
        print("尝试使用静态渲染模式")
        env.render(mode='human')
    
    print("\n渲染完成！")
    
    return madqn


def main():
    """
    主函数：可以选择训练或测试MADQN模型
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MADQN实验脚本')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'both'],
                        help='运行模式: train(训练), test(测试), both(训练后测试)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_madqn_model()
    elif args.mode == 'test':
        test_madqn_model()
    elif args.mode == 'both':
        madqn = train_madqn_model()
        if madqn is not None:
            # 重新创建环境用于测试
            env = Env()
            # 设置epsilon为最小值用于测试
            for agent_id in range(madqn.num_agents):
                madqn.epsilons[agent_id] = madqn.epsilon_min
            
            print("\n" + "=" * 50)
            print("训练完成后立即测试")
            print("=" * 50)
            run_with_agent(env, madqn, max_steps=2000, training=False, verbose=True)
            
            # 渲染
            print("\n渲染动画...")
            os.makedirs("results", exist_ok=True)
            gif_path = os.path.join("results", "madqn_trained_test.gif")
            try:
                env.render_animation(interval=1, save_path=gif_path)
                print(f"动画已保存到: {gif_path}")
            except Exception as e:
                print(f"动画渲染出错: {e}")


if __name__ == '__main__':
    main()
