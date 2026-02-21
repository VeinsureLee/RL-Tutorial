"""
训练模块：DQN/MADQN 训练逻辑，入口为 main()。
调试请在 rl_algorithms 目录下运行本文件，或从项目根目录运行 python -m rl_algorithms.train。
"""
import sys
import os

# 保证从项目根目录可导入 env、utils
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from tqdm import tqdm

from env.env import Env
from rl_algorithms.rl.dqn import DQN
from rl_algorithms.rl.madqn import MADQN
from rl_algorithms.plot.plot import plot_dqn, plot_madqn, FIG_DIR
from utils.logger_handler import get_logger


def train_dqn(env, dqn):
    """训练 DQN，返回 (dqn, return_list)。"""
    epsilon = dqn.epsilon
    return_list = []
    print(f"Begin to train DQN, iteration: {dqn.iteration}")
    for i in range(dqn.iteration):
        dqn.epsilon = epsilon
        pbar = tqdm(range(1, dqn.num_episodes + 1),
                    desc=f"Iteration({i+1}) progress", unit="episode")
        for ep in pbar:
            states, _ = env.reset()
            state = states[dqn.agent_id]
            ep_return = 0
            for t in range(dqn.episode_length):
                action_idx = dqn.take_action(state, training=True)
                action = env.action_space[action_idx]
                next_state, reward, done, _ = env.step(action)
                ep_return += reward
                dqn.buffer.add(state, action_idx, reward, next_state, done)
                if len(dqn.buffer) >= dqn.mini_batch_size:
                    dqn.update()
                state = next_state
                if t % dqn.update_freq == 0:
                    dqn.update_target_qnet()
                if done:
                    break
            return_list.append(ep_return)
            dqn.epsilon = max(dqn.epsilon_min, dqn.epsilon * dqn.epsilon_decay)
            pbar.set_postfix({
                'Episode': ep,
                'Return': f'{ep_return:.2f}',
                'Epsilon': f'{dqn.epsilon:.3f}'
            })
    return dqn, return_list


def train_madqn(env, madqn):
    """训练 MADQN，返回 (madqn, return_list, agent_return_lists, ber_list, agent_ber_lists)。"""
    logger = get_logger("madqn")  # 单一日志：控制台 INFO，文件 DEBUG（含每 episode 详情）

    logger.info("")
    logger.info("#" * 80)
    logger.info("#                          MADQN 训练开始                                       #")
    logger.info("#" * 80)
    logger.info("训练参数: iteration=%s, num_episodes=%s, episode_length=%s, num_agents=%s, lr=%s, gamma=%s, "
                "epsilon=%s, epsilon_min=%s, epsilon_decay=%s, mini_batch_size=%s, update_freq=%s, device=%s",
                madqn.iteration, madqn.num_episodes, madqn.episode_length, madqn.num_agents,
                madqn.lr, madqn.gamma, madqn.epsilon, madqn.epsilon_min, madqn.epsilon_decay,
                madqn.mini_batch_size, madqn.update_freq, madqn.device)
    print(f"Begin to train MADQN, iteration: {madqn.iteration}")

    epsilon = madqn.epsilon
    return_list = []
    agent_return_lists = [[] for _ in range(madqn.num_agents)]
    ber_list = []
    agent_ber_lists = [[] for _ in range(madqn.num_agents)]

    for i in range(madqn.iteration):
        madqn.epsilon = epsilon
        logger.info("---------- Iteration %s/%s 开始 ----------", i + 1, madqn.iteration)
        pbar = tqdm(range(1, madqn.num_episodes + 1),
                    desc=f"Iteration({i+1}) progress", unit="episode")
        for ep in pbar:
            states, _ = env.reset()
            if env.num_agents == 1:
                states = [states]

            agent_returns = [0.0] * madqn.num_agents
            agent_arrived = [False] * madqn.num_agents
            episode_ber_sum = 0.0
            episode_ber_sum_per_agent = [0.0] * madqn.num_agents
            step_count = 0

            for t in range(madqn.episode_length):
                action_indices = madqn.take_action(states, training=True)
                actions = [env.action_space[idx] for idx in action_indices]
                step_result = env.step(actions)
                next_states, rewards, dones, info = step_result
                ber_this = info.get("ber", info.get("ber_per_agent", [0.5] * madqn.num_agents))
                episode_ber_sum += np.mean(ber_this)
                for j in range(madqn.num_agents):
                    episode_ber_sum_per_agent[j] += ber_this[j]
                step_count += 1

                if env.num_agents == 1:
                    next_states = [next_states]
                    rewards = [rewards]
                    dones = [dones]

                for agent_id in range(madqn.num_agents):
                    if not agent_arrived[agent_id]:
                        agent_returns[agent_id] += rewards[agent_id]
                    if dones[agent_id]:
                        agent_arrived[agent_id] = True

                ep_return = sum(agent_returns)
                for agent_id in range(madqn.num_agents):
                    madqn.buffer[agent_id].add(
                        states[agent_id], action_indices[agent_id],
                        rewards[agent_id], next_states[agent_id], dones[agent_id])
                    if len(madqn.buffer[agent_id]) >= madqn.mini_batch_size:
                        madqn.update(agent_id)

                states = next_states
                if t % madqn.update_freq == 0:
                    for agent_id in range(madqn.num_agents):
                        madqn.update_target_qnet(agent_id)
                if all(dones):
                    break

            return_list.append(ep_return)
            for agent_id in range(madqn.num_agents):
                agent_return_lists[agent_id].append(agent_returns[agent_id])
            episode_avg_ber = episode_ber_sum / step_count if step_count > 0 else 0.0
            ber_list.append(episode_avg_ber)
            for j in range(madqn.num_agents):
                agent_avg_ber = episode_ber_sum_per_agent[j] / step_count if step_count > 0 else 0.0
                agent_ber_lists[j].append(agent_avg_ber)

            madqn.epsilon = max(madqn.epsilon_min, madqn.epsilon * madqn.epsilon_decay)
            return_str = ', '.join([f'A{j+1}:{r:.2f}' for j, r in enumerate(agent_returns)])
            agent_ber_str = ', '.join([f'A{j+1}:{agent_ber_lists[j][-1]:.4f}' for j in range(madqn.num_agents)])
            logger.debug(
                "Iter %s Ep %s | TotalReturn=%.2f | AgentReturns=[%s] | AvgBER=%.4f [%s] | Epsilon=%.3f",
                i + 1, ep, ep_return, return_str, episode_avg_ber, agent_ber_str, madqn.epsilon)
            pbar.set_postfix({
                'Episode': ep, 'Total Return': f'{ep_return:.2f}',
                'Agent Returns': return_str, 'Epsilon': f'{madqn.epsilon:.3f}'
            })

    if return_list:
        avg_ret = sum(return_list) / len(return_list)
        last10 = sum(return_list[-10:]) / min(10, len(return_list))
        logger.info("MADQN 训练结束  总episode数=%s  平均Return=%.2f  最后10ep平均Return=%.2f",
                    len(return_list), avg_ret, last10)
    return madqn, return_list, agent_return_lists, ber_list, agent_ber_lists


def main():
    """调试入口：可选运行 DQN 或 MADQN，训练后保存模型并绘图到 rl_algorithms/figs。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env()

    # 选择算法: "dqn" 或 "madqn"
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


if __name__ == "__main__":
    main()
