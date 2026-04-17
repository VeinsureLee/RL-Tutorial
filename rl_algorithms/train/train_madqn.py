"""MADQN 训练逻辑：外层 iteration + 内层 episode，每轮 iteration 重置 epsilon。"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from tqdm import tqdm
from utils.logger_handler import get_logger


def train_madqn(env, madqn, num_iterations=10, num_episodes=200, episode_length=5000,
                batch_size=128, train_interval=4):
    """训练 MADQN，返回 (madqn, return_list, agent_return_lists, ber_list, agent_ber_lists)。
    外层 iteration 循环：每轮重置 epsilon 到初始值再衰减，避免局部最优。
    """
    logger = get_logger("madqn")
    logger.info("")
    logger.info("#" * 80)
    logger.info("#                          MADQN 训练开始                                       #")
    logger.info("#" * 80)
    logger.info("训练参数: num_iterations=%s, num_episodes=%s, episode_length=%s, num_agents=%s, "
                "lr=%.6f, gamma=%.2f, epsilon=%.2f, batch_size=%s, update_freq=%s",
                num_iterations, num_episodes, episode_length, madqn.num_agents,
                madqn.optimizers[0].param_groups[0]['lr'], madqn.gamma,
                madqn.epsilon, batch_size, madqn.update_freq)

    madqn.batch_size = batch_size
    return_list = []
    agent_return_lists = [[] for _ in range(madqn.num_agents)]
    ber_list = []
    agent_ber_lists = [[] for _ in range(madqn.num_agents)]
    global_step = 0
    epsilon_start = madqn.epsilon

    total_episodes = num_iterations * num_episodes
    ep_count = 0

    for it in range(1, num_iterations + 1):
        madqn.epsilon = epsilon_start

        pbar = tqdm(range(1, num_episodes + 1),
                    desc=f"MADQN Iter {it}/{num_iterations}",
                    unit="ep")
        for ep in pbar:
            ep_count += 1
            states = env.reset()
            ep_return = np.zeros(madqn.num_agents)
            ep_ber = []

            for step in range(episode_length):
                actions = madqn.take_action(states, training=True)
                next_states, rewards, dones, info = env.step(actions)

                for i in range(madqn.num_agents):
                    madqn.buffers[i].add(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    ep_return[i] += rewards[i]

                global_step += 1
                if global_step % train_interval == 0:
                    for i in range(madqn.num_agents):
                        if len(madqn.buffers[i]) >= batch_size:
                            batch = madqn.buffers[i].sample(batch_size)
                            madqn.update(i, batch)

                if global_step % madqn.update_freq == 0:
                    for i in range(madqn.num_agents):
                        madqn.update_target_qnet(i)

                ep_ber.append(float(info["ber"].mean()))
                states = next_states

                if env.all_done:
                    break

            return_list.append(float(ep_return.sum()))
            for i in range(madqn.num_agents):
                agent_return_lists[i].append(float(ep_return[i]))
            avg_ber = np.mean(ep_ber) if ep_ber else 0.0
            ber_list.append(avg_ber)
            for i in range(madqn.num_agents):
                agent_ber_lists[i].append(float(info["ber"][i]))

            madqn.epsilon = max(madqn.epsilon_min, madqn.epsilon * madqn.epsilon_decay)

            return_str = ', '.join([f'A{j+1}:{r:.1f}' for j, r in enumerate(ep_return)])
            logger.debug("Iter %s Ep %s/%s | Return=%.1f | [%s] | BER=%.6f | eps=%.3f",
                         it, ep, num_episodes, ep_return.sum(), return_str, avg_ber, madqn.epsilon)
            pbar.set_postfix({
                'R': f'{ep_return.sum():.1f}',
                'BER': f'{avg_ber:.4f}',
                'eps': f'{madqn.epsilon:.3f}',
                'total': f'{ep_count}/{total_episodes}',
            })

    if return_list:
        avg_ret = sum(return_list) / len(return_list)
        last10 = sum(return_list[-10:]) / min(10, len(return_list))
        logger.info("MADQN 训练结束  总episode数=%s  平均Return=%.2f  最后10ep平均Return=%.2f",
                    len(return_list), avg_ret, last10)

    return madqn, return_list, agent_return_lists, ber_list, agent_ber_lists
