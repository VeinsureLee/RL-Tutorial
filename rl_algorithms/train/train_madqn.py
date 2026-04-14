"""MADQN 训练逻辑。"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from tqdm import tqdm
from env.env import Env
from rl_algorithms.structure.madqn import MADQN
from utils.logger_handler import get_logger


def train_madqn(env, madqn):
    """训练 MADQN，返回 (madqn, return_list, agent_return_lists, ber_list, agent_ber_lists)。"""
    logger = get_logger("madqn")
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
        pbar = tqdm(range(1, madqn.num_episodes + 1), desc=f"Iteration({i+1}) progress", unit="episode")
        for ep in pbar:
            states, _ = env.reset()  # 始终返回 list of tuples

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
                    next_states, rewards, dones = [next_states], [rewards], [dones]

                for agent_id in range(madqn.num_agents):
                    if not agent_arrived[agent_id]:
                        agent_returns[agent_id] += rewards[agent_id]
                    if dones[agent_id]:
                        agent_arrived[agent_id] = True

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

            return_list.append(sum(agent_returns))
            for agent_id in range(madqn.num_agents):
                agent_return_lists[agent_id].append(agent_returns[agent_id])
            episode_avg_ber = episode_ber_sum / step_count if step_count > 0 else 0.0
            ber_list.append(episode_avg_ber)
            for j in range(madqn.num_agents):
                agent_ber_lists[j].append(episode_ber_sum_per_agent[j] / step_count if step_count > 0 else 0.0)

            madqn.epsilon = max(madqn.epsilon_min, madqn.epsilon * madqn.epsilon_decay)
            return_str = ', '.join([f'A{j+1}:{r:.2f}' for j, r in enumerate(agent_returns)])
            logger.debug("Iter %s Ep %s | TotalReturn=%.2f | AgentReturns=[%s] | Epsilon=%.3f",
                         i + 1, ep, sum(agent_returns), return_str, madqn.epsilon)
            pbar.set_postfix({'Episode': ep, 'Total Return': f'{sum(agent_returns):.2f}', 'Agent Returns': return_str, 'Epsilon': f'{madqn.epsilon:.3f}'})

    if return_list:
        avg_ret = sum(return_list) / len(return_list)
        last10 = sum(return_list[-10:]) / min(10, len(return_list))
        logger.info("MADQN 训练结束  总episode数=%s  平均Return=%.2f  最后10ep平均Return=%.2f",
                    len(return_list), avg_ret, last10)
    return madqn, return_list, agent_return_lists, ber_list, agent_ber_lists
