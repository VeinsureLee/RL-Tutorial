import sys
import os
import numpy as np
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.env import Env
from rl_algorithms.agent import Agent
from rl_algorithms.dqn import DQN


def run_with_agent(env, agent, max_steps=200, training=False, verbose=True):
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
    
    for step in range(max_steps):
        # 使用agent选择动作
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
        
        # 如果是DQN且处于训练模式，存储经验并训练
        if training and isinstance(agent, DQN):
            agent.remember(states, actions, rewards, next_states, dones)
            loss = agent.train()
            if loss is not None and step % 10 == 0 and verbose:
                print(f"训练损失: {loss:.4f}")
        
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


def train_dqn(env, dqn, num_episodes=500, max_steps_per_episode=500, verbose=True):
    """
    训练DQN模型
    :param env: 环境实例
    :param dqn: DQN实例
    :param num_episodes: 训练轮数
    :param max_steps_per_episode: 每轮最大步数
    :param verbose: 是否打印详细信息
    :return: 训练历史
    """
    success_count = 0
    episode_rewards = []
    episode_lengths = []
    
    print("=" * 50)
    print("开始训练DQN")
    print("=" * 50)
    
    for episode in range(num_episodes):
        states, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # 选择动作
            actions = dqn.select_action(states, training=True)
            
            # 确保actions是列表格式
            if not isinstance(actions, list):
                if isinstance(actions, tuple) and len(actions) == 2:
                    # 检查是否是动作tuple（两个数字）还是其他
                    if isinstance(actions[0], (int, float, np.integer, np.floating)):
                        actions = [actions]
                    else:
                        actions = list(actions)
                else:
                    actions = [actions]
            
            # 确保actions数量正确
            if len(actions) != env.num_agents:
                if env.num_agents == 1:
                    actions = [actions[0] if len(actions) > 0 else random.choice(env.action_space)]
                else:
                    actions = [random.choice(env.action_space) for _ in range(env.num_agents)]
            
            # 执行动作
            result = env.step(actions)
            next_states, rewards, dones, _ = result
            
            # 确保返回的是列表格式
            if not isinstance(next_states, list):
                next_states = [next_states]
            if not isinstance(rewards, list):
                rewards = [rewards]
            if not isinstance(dones, list):
                dones = [dones]
            
            # 存储经验并训练
            dqn.remember(states, actions, rewards, next_states, dones)
            loss = dqn.train()
            
            episode_reward += sum(rewards)
            episode_length = step + 1
            states = next_states
            
            # 检查是否完成
            if all(dones):
                success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            success_rate = success_count / (episode + 1) * 100
            print(f"Episode {episode+1}/{num_episodes} - "
                  f"成功率: {success_rate:.1f}% - "
                  f"平均奖励: {avg_reward:.2f} - "
                  f"平均步数: {avg_length:.1f} - "
                  f"Epsilon: {dqn.agents[0].epsilon:.3f}")
    
    print(f"\n训练完成！总成功率: {success_count/num_episodes*100:.1f}%")
    return episode_rewards, episode_lengths


def main():
    """
    主函数：创建环境和Agent，运行Agent策略并渲染
    """
    print("=" * 50)
    print("DQN训练和测试")
    print("=" * 50)
    
    # 创建环境
    env = Env()
    
    # 创建DQN Agent
    print("\n创建DQN Agent...")
    dqn = DQN(env, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
              epsilon_decay=0.995, memory_size=10000, batch_size=64, hidden_dim=128)
    
    # 训练DQN
    print("\n开始训练...")
    train_dqn(env, dqn, num_episodes=200, max_steps_per_episode=500, verbose=True)
    
    # 测试训练后的模型（设置epsilon为最小值，使用学习到的策略）
    print("\n" + "=" * 50)
    print("测试训练后的DQN模型")
    print("=" * 50)
    
    # 将epsilon设置为最小值，确保使用学习到的策略
    for agent in dqn.agents:
        agent.epsilon = agent.epsilon_min
    
    # 重置环境用于测试
    test_env = Env()
    success = run_with_agent(test_env, dqn, max_steps=1000, training=False, verbose=True)
    
    print("\n" + "=" * 50)
    print("开始渲染环境")
    print("=" * 50)
    
    # 渲染静态图像
    print("\n渲染静态图像...")
    test_env.render(mode='human')
    
    # 渲染动画
    print("\n渲染动画...")
    try:
        test_env.render_animation(interval=100, save_path=None)
    except Exception as e:
        print(f"动画渲染出错: {e}")
        print("尝试使用静态渲染模式")
        test_env.render(mode='human')
    
    print("\n渲染完成！")
    
    # 可选：保存模型
    dqn.save("models/dqn_model")


if __name__ == '__main__':
    main()

