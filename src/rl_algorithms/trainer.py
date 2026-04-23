"""
统一训练接口：DQN 与 MADQN 复用同一循环。
对外只暴露 train(env, model, **kwargs) -> dict 一个函数。

奖励成分已从 env.step().info 中按 4 路返回：
    step_rewards, approach_rewards, comm_rewards；total = 三者之和（即 env.step 返回的 rewards）。
"""
import numpy as np
from tqdm import tqdm

from rl_algorithms.algorithms import DQN, MADQN


def _take_all_actions(env, model, states, training: bool):
    """按模型类型返回所有 agent 的动作列表。DQN 下非 agent_id 的 agent 走均匀随机。"""
    if isinstance(model, MADQN):
        return model.take_action(states, training=training)
    actions = [int(np.random.randint(env.n_actions)) for _ in range(env.num_agents)]
    actions[model.agent_id] = model.take_action(states[model.agent_id], training=training)
    return actions


def _push_and_maybe_update(model, step_data, global_step: int, train_interval: int):
    """把 (s, a, r, ns, d) 写进 replay；每 train_interval 步采样一次进行梯度更新。"""
    states, actions, rewards, next_states, dones = step_data
    if isinstance(model, MADQN):
        for i in range(model.num_agents):
            model.buffers[i].add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        if global_step % train_interval == 0:
            for i in range(model.num_agents):
                if len(model.buffers[i]) >= model.batch_size:
                    model.update(i, model.buffers[i].sample(model.batch_size))
    else:
        i = model.agent_id
        model.buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        if global_step % train_interval == 0 and len(model.buffer) >= model.batch_size:
            model.update(model.buffer.sample(model.batch_size))


def _sync_target(model, global_step: int):
    if global_step % model.update_freq == 0:
        if isinstance(model, MADQN):
            for i in range(model.num_agents):
                model.update_target_qnet(i)
        else:
            model.update_target_qnet()


def train(env, model,
          num_iterations: int = 5,
          num_episodes: int = 50,
          episode_length: int = 5000,
          batch_size: int = 128,
          train_interval: int = 4,
          logger=None) -> dict:
    """
    训练入口。返回 dict，key:
        return_list, step_return_list, approach_return_list, comm_return_list (每 ep 所有 agent 求和)
        agent_return_lists / agent_step_return_lists / agent_approach_return_lists / agent_comm_return_lists
        ber_list           : 每 ep 在所有 step/agent 上的 mean(-log10 BER)（已裁剪下限 1e-20）
        agent_ber_lists    : 每 ep 每 agent 的 mean(-log10 BER)
    """
    model.batch_size = batch_size
    num_agents = env.num_agents
    is_madqn = isinstance(model, MADQN)
    algo_name = "MADQN" if is_madqn else "DQN"

    return_list = []
    step_return_list = []
    approach_return_list = []
    comm_return_list = []
    agent_return_lists = [[] for _ in range(num_agents)]
    agent_step_return_lists = [[] for _ in range(num_agents)]
    agent_approach_return_lists = [[] for _ in range(num_agents)]
    agent_comm_return_lists = [[] for _ in range(num_agents)]
    ber_list = []
    agent_ber_lists = [[] for _ in range(num_agents)]

    global_step = 0
    epsilon_start = model.epsilon
    total_eps = num_iterations * num_episodes
    ep_counter = 0

    if logger:
        logger.info("%s training start: iters=%s episodes=%s ep_len=%s agents=%s lr=%.1e gamma=%.2f eps=%.2f bs=%s",
                    algo_name, num_iterations, num_episodes, episode_length, num_agents,
                    (model.optimizers[0].param_groups[0]['lr'] if is_madqn else model.optimizer.param_groups[0]['lr']),
                    model.gamma, model.epsilon, batch_size)

    for it in range(1, num_iterations + 1):
        # 每轮外层迭代重置 epsilon，借助噪声跳出局部最优
        model.epsilon = epsilon_start
        pbar = tqdm(range(1, num_episodes + 1),
                    desc=f"{algo_name} iter {it}/{num_iterations}", unit="ep")
        for ep in pbar:
            ep_counter += 1
            states = env.reset()

            ep_total = np.zeros(num_agents)
            ep_step = np.zeros(num_agents)
            ep_approach = np.zeros(num_agents)
            ep_comm = np.zeros(num_agents)
            # 收集 BER 的负对数：mean(-log10(ber)) 更能反映典型通信质量
            ep_neglog_ber_per_agent = [[] for _ in range(num_agents)]

            for _ in range(episode_length):
                actions = _take_all_actions(env, model, states, training=True)
                next_states, rewards, dones, info = env.step(actions)

                _push_and_maybe_update(
                    model, (states, actions, rewards, next_states, dones),
                    global_step + 1, train_interval,
                )
                global_step += 1
                _sync_target(model, global_step)

                for i in range(num_agents):
                    ep_total[i] += rewards[i]
                    ep_step[i] += info["step_rewards"][i]
                    ep_approach[i] += info["approach_rewards"][i]
                    ep_comm[i] += info["comm_rewards"][i]
                    # BER 下限裁剪后取 -log10
                    b = max(float(info["ber"][i]), 1e-20)
                    ep_neglog_ber_per_agent[i].append(-np.log10(b))

                states = next_states
                if env.all_done:
                    break

            # episode 汇总
            return_list.append(float(ep_total.sum()))
            step_return_list.append(float(ep_step.sum()))
            approach_return_list.append(float(ep_approach.sum()))
            comm_return_list.append(float(ep_comm.sum()))
            ep_neglog_all = []
            for i in range(num_agents):
                agent_return_lists[i].append(float(ep_total[i]))
                agent_step_return_lists[i].append(float(ep_step[i]))
                agent_approach_return_lists[i].append(float(ep_approach[i]))
                agent_comm_return_lists[i].append(float(ep_comm[i]))
                mean_neglog_i = float(np.mean(ep_neglog_ber_per_agent[i])) if ep_neglog_ber_per_agent[i] else 0.0
                agent_ber_lists[i].append(mean_neglog_i)
                ep_neglog_all.extend(ep_neglog_ber_per_agent[i])
            ber_list.append(float(np.mean(ep_neglog_all)) if ep_neglog_all else 0.0)

            # epsilon 衰减
            model.epsilon = max(model.epsilon_min, model.epsilon * model.epsilon_decay)

            if logger:
                logger.debug("iter %s ep %s/%s R=%.1f step=%.1f approach=%.1f comm=%.1f -log10BER=%.2f eps=%.3f",
                             it, ep, num_episodes, ep_total.sum(), ep_step.sum(),
                             ep_approach.sum(), ep_comm.sum(), ber_list[-1], model.epsilon)
            pbar.set_postfix({
                "R": f"{ep_total.sum():.1f}",
                "appr": f"{ep_approach.sum():.1f}",
                "comm": f"{ep_comm.sum():.1f}",
                "-logBER": f"{ber_list[-1]:.1f}",
                "eps": f"{model.epsilon:.3f}",
                "ep": f"{ep_counter}/{total_eps}",
            })

    if return_list and logger:
        avg = sum(return_list) / len(return_list)
        last10 = sum(return_list[-10:]) / min(10, len(return_list))
        logger.info("%s training done: total_eps=%s avg_return=%.2f last10_avg_return=%.2f",
                    algo_name, len(return_list), avg, last10)

    return {
        "return_list": return_list,
        "step_return_list": step_return_list,
        "approach_return_list": approach_return_list,
        "comm_return_list": comm_return_list,
        "agent_return_lists": agent_return_lists,
        "agent_step_return_lists": agent_step_return_lists,
        "agent_approach_return_lists": agent_approach_return_lists,
        "agent_comm_return_lists": agent_comm_return_lists,
        "ber_list": ber_list,
        "agent_ber_lists": agent_ber_lists,
    }
