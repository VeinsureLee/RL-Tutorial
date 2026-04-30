"""统统一训练接口：DQN / MADQN / QMIX / VDN / PPO / MAPPO 复用同一循环。

奖励成分已从 env.step().info 中按 4 路返回：
    step_rewards, approach_rewards, comm_rewards；total = 三者之和（即 env.step 返回的 rewards）。

history 还包含 time_list：每 episode 的 wall-clock 耗时（秒），用于 time.png 与 scalability 分析。
"""
import time
import numpy as np
from tqdm import tqdm

from rl_algorithms.dqn import DQN
from rl_algorithms.madqn import MADQN
from rl_algorithms.qmix import QMIX
from rl_algorithms.vdn import VDN
from rl_algorithms.ppo import PPO
from rl_algorithms.mappo import MAPPO

# 在文件开头导入 torch，避免循环导入问题
import torch


def _is_on_policy(model):
    """判断是否为 on-policy 算法（PPO/MAPPO）。"""
    return isinstance(model, (PPO, MAPPO))


def _take_all_actions(env, model, states, training: bool):
    """按模型类型返回所有 agent 的动作列表。DQN 下非 agent_id 的 agent 走均匀随机。"""
    if isinstance(model, (MADQN, QMIX, VDN)):
        return model.take_action(states, training=training)
    if isinstance(model, MAPPO):
        return model.take_action(states, training=training)
    if isinstance(model, PPO):
        # PPO: 只训练 agent_id，其它给随机动作
        actions = [int(np.random.randint(env.n_actions)) for _ in range(env.num_agents)]
        if training:
            ppo_action, value, log_prob = model.take_action(states[model.agent_id], training=training)
            actions[model.agent_id] = ppo_action
            return actions, value, log_prob
        else:
            actions[model.agent_id] = model.take_action(states[model.agent_id], training=training)
            return actions
    # DQN / SharedMADQN
    actions = [int(np.random.randint(env.n_actions)) for _ in range(env.num_agents)]
    actions[model.agent_id] = model.take_action(states[model.agent_id], training=training)
    return actions


def _push_and_maybe_update_off_policy(model, step_data, global_step: int, train_interval: int):
    """off-policy 算法（DQN/MADQN/QMIX/VDN）：push 到 replay 并按间隔更新。"""
    states, actions, rewards, next_states, dones = step_data
    if isinstance(model, (QMIX, VDN)):
        # QMIX / VDN：单一联合 buffer，单次联合反传
        model.buffer.add(states, actions, rewards, next_states, dones)
        if global_step % train_interval == 0 and len(model.buffer) >= model.batch_size:
            model.update(model.buffer.sample(model.batch_size))
    elif isinstance(model, MADQN):
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


def _push_on_policy_data(model, step_data, next_states, dones, prev_dones=None):
    """on-policy 算法（PPO/MAPPO）：push 到 rollout buffer。

    prev_dones：进入本 step 之前各 agent 的 done 状态。已经 done 的 agent 不再 push
    任何转移，避免 (action=0, value=0, log_prob=0) 的伪样本污染 critic 与 importance ratio。
    """
    states, actions, rewards, extra = step_data
    if isinstance(model, MAPPO):
        actions_list, values_list, log_probs_list = extra
        # 获取全局信息
        if hasattr(model, '_cached_all_starts') and model._cached_all_starts is not None:
            all_starts = model._cached_all_starts
            all_targets = model._cached_all_targets
        else:
            all_starts = np.array(model.current_start_indices, dtype=np.int64)
            all_targets = np.array([
                model.env.pos_to_index(*model.env.target_states[i])
                for i in range(model.num_agents)
            ], dtype=np.int64)
        all_states = np.array(states, dtype=np.int64)

        # reward_scale 把 env 的原始 reward (range [-15, +6]/step, episode return ~ -10000~+50)
        # 压到 ~O(1) 量级，让 PPO 的 value-loss / value-clip(epsilon=0.15) / Adam step
        # 落入它们假设的工作区间。注意 value 不需要在此缩放：critic 学的就是 scaled return，
        # 它在 collection 时输出的 value 已经在 scaled 单位上，与 scaled reward 自洽。
        scale = model.reward_scale
        for i in range(model.num_agents):
            if prev_dones is not None and prev_dones[i]:
                continue  # 跳过已 done agent 的伪转移
            target_idx = model.env.pos_to_index(*model.env.target_states[i])
            start_idx = model.current_start_indices[i]
            model.buffers[i].add(
                states[i], actions_list[i], rewards[i] * scale, values_list[i], log_probs_list[i],
                int(start_idx), int(target_idx), i, dones[i],
                all_states, all_starts, all_targets
            )
    elif isinstance(model, PPO):
        actions_list, value, log_prob = extra
        i = model.agent_id
        target_idx = model.env.pos_to_index(*model.env.target_states[i])
        start_idx = model.current_start_idx
        model.buffer.add(
            states[i], actions_list[i], rewards[i], value, log_prob,
            int(start_idx), int(target_idx), dones[i]
        )


def _update_on_policy(model, next_states, dones):
    """on-policy 算法（PPO/MAPPO）：执行 rollout 后的更新。"""
    if isinstance(model, MAPPO):
        # 准备 next 的全局信息（critic 输入）
        next_all_states = np.array(next_states, dtype=np.int64)
        next_all_starts = np.array(model.current_start_indices, dtype=np.int64)
        next_all_targets = np.array([
            model.env.pos_to_index(*model.env.target_states[i])
            for i in range(model.num_agents)
        ], dtype=np.int64)

        # 共享 critic：对所有未 done 的 agent 来说 next_value 相同（同一个全局状态、同一份参数），
        # 只算一次；done 的 agent 用 0 bootstrap。
        flags = model.env.done_flags
        any_alive = flags is None or any(not f for f in flags)
        if any_alive:
            next_all_states_t = torch.tensor(next_all_states, dtype=torch.long, device=model.device).unsqueeze(0)
            next_all_starts_t = torch.tensor(next_all_starts, dtype=torch.long, device=model.device).unsqueeze(0)
            next_all_targets_t = torch.tensor(next_all_targets, dtype=torch.long, device=model.device).unsqueeze(0)
            next_all_agent_idx_t = torch.tensor(list(range(model.num_agents)), dtype=torch.long, device=model.device).unsqueeze(0)
            with torch.no_grad():
                shared_next_v = float(model._shared_net.forward_critic(
                    next_all_states_t, next_all_starts_t, next_all_targets_t, next_all_agent_idx_t
                ).item())
        else:
            shared_next_v = 0.0
        next_values = [
            0.0 if (flags is not None and flags[i]) else shared_next_v
            for i in range(model.num_agents)
        ]
        return model.update_all(next_values)
    elif isinstance(model, PPO):
        i = model.agent_id
        if model.env.done_flags is not None and model.env.done_flags[i]:
            next_value = 0.0
        else:
            target_idx = model.env.pos_to_index(*model.env.target_states[i])
            start_idx = model.current_start_idx
            s = torch.tensor([next_states[i]], dtype=torch.long, device=model.device)
            t = torch.tensor([target_idx], dtype=torch.long, device=model.device)
            start = torch.tensor([start_idx], dtype=torch.long, device=model.device)
            with torch.no_grad():
                _, next_value = model.net(s, t, start)
            next_value = float(next_value.item())
        return model.update(next_value)
    return 0.0


def _sync_target(model, global_step: int):
    if _is_on_policy(model):
        return
    if global_step % model.update_freq == 0:
        if isinstance(model, (QMIX, VDN)):
            model.update_target_qnet()  # 一次同步所有 Q_i（QMIX 含 mixer）
        elif isinstance(model, MADQN):
            for i in range(model.num_agents):
                model.update_target_qnet(i)
        else:
            model.update_target_qnet()


def train(env, model,
          num_iterations: int = 5,
          num_episodes: int = 50,
          episode_length: int = 5000,
          batch_size: int = 64,
          train_interval: int = 4,
          update_interval: int = 2048,  # on-policy: 每多少步更新一次
          logger=None) -> dict:
    """
    训练入口。返回 dict，key:
        return_list, step_return_list, approach_return_list, comm_return_list (每 ep 所有 agent 求和)
        agent_return_lists / agent_step_return_lists / agent_approach_return_lists / agent_comm_return_lists
        ber_list           : 每 ep 在所有 step/agent 上的 mean(-log10 BER)（已裁剪下限 1e-20）
        agent_ber_lists    : 每 ep 每 agent 的 mean(-log10 BER)
        ber_max_list       : 每 ep 在所有 step/agent 上的 max(-log10 BER) ≡ -log10(min BER)
        agent_ber_max_lists: 每 ep 每 agent 的 max(-log10 BER)
    """
    model.batch_size = batch_size
    num_agents = env.num_agents
    is_madqn = isinstance(model, MADQN)
    is_qmix = isinstance(model, QMIX)
    is_vdn = isinstance(model, VDN)
    is_ppo = isinstance(model, PPO)
    is_mappo = isinstance(model, MAPPO)
    is_on_policy = _is_on_policy(model)

    if is_qmix:
        algo_name = "QMIX"
    elif is_vdn:
        algo_name = "VDN"
    elif is_madqn:
        algo_name = "MADQN"
    elif is_ppo:
        algo_name = "PPO"
    elif is_mappo:
        algo_name = "MAPPO"
    else:
        algo_name = "DQN"

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
    # 每 ep 的最佳 BER 质量：max(-log10 BER) ≡ -log10(min BER)
    ber_max_list = []
    agent_ber_max_lists = [[] for _ in range(num_agents)]
    time_list = []  # 每 episode wall-clock 耗时（秒）
    reached_list = []  # 每 episode 到达目标的 agent 数（0..num_agents）

    global_step = 0
    # 记录 iter 开始时的探索强度，用于每 iter 重置。
    # off-policy: epsilon 控制"采样"（不改网络参数），iter 重置 = 注入探索样本，无破坏性。
    # on-policy : entropy_coef 是 loss 里的正则项系数（直接反传到参数）；重置会把已经
    #             锐化的 actor 拉回均匀分布，破坏前一个 iter 学到的策略。所以 on-policy
    #             不重置 entropy_coef，让它在整个 num_iterations*num_episodes 中单调衰减。
    #             此时 iter 外层循环对 on-policy 仅起进度条/日志分组作用。
    if not is_on_policy:
        epsilon_start = model.epsilon
    total_eps = num_iterations * num_episodes
    ep_counter = 0


    if logger:
        if is_madqn:
            lr_val = model.optimizers[0].param_groups[0]['lr']
        elif is_mappo:
            lr_val = model.optimizers[0].param_groups[0]['lr']
        else:
            # DQN / PPO / QMIX / VDN 都只有一个 optimizer
            lr_val = model.optimizer.param_groups[0]['lr']
        logger.info("%s training start: iters=%s episodes=%s ep_len=%s agents=%s lr=%.1e",
                    algo_name, num_iterations, num_episodes, episode_length, num_agents, lr_val)
        if is_on_policy:
            logger.info("on-policy: update_interval=%d steps", update_interval)
        else:
            logger.info("off-policy: gamma=%.2f eps=%.2f batch=%d train_interval=%d",
                        model.gamma, model.epsilon, batch_size, train_interval)

    for it in range(1, num_iterations + 1):
        # 仅 off-policy 在 iter 开头重置 epsilon —— 让每轮重新经历"广泛探索 → 接近确定"。
        # on-policy 不重置 entropy_coef（理由见上方注释），iter 仅作展示分组。
        if not is_on_policy:
            model.epsilon = epsilon_start
        pbar = tqdm(range(1, num_episodes + 1),
                    desc=f"{algo_name} iter {it}/{num_iterations}",
                    unit="ep", position=0, leave=True, dynamic_ncols=True)
        # 第二行只显示 desc 的轻量 bar，用于扩展信息（time / approach / comm / -logBER / eps 等）
        detail_bar = tqdm(bar_format="{desc}", position=1, leave=False, dynamic_ncols=True)
        for ep in pbar:
            ep_counter += 1
            ep_t0 = time.perf_counter()  # 每 episode 用 wall-clock 计时
            states = env.reset()

            # on-policy 算法在 reset 时更新 start_idx
            if is_on_policy:
                if is_mappo:
                    model.reset_start_indices()
                elif is_ppo:
                    model.reset_start_idx()

            ep_total = np.zeros(num_agents)
            ep_step = np.zeros(num_agents)
            ep_approach = np.zeros(num_agents)
            ep_comm = np.zeros(num_agents)
            # 收集 BER 的负对数：mean(-log10(ber)) 更能反映典型通信质量
            ep_neglog_ber_per_agent = [[] for _ in range(num_agents)]

            for step_in_ep in range(episode_length):
                # 进入本 step 之前各 agent 的 done 状态：用于过滤 post-done 垃圾转移
                prev_dones = env.done_flags.copy()

                action_output = _take_all_actions(env, model, states, training=True)

                if is_on_policy:
                    # on-policy: action_output = (actions_list, extra) 或 actions_list
                    if is_mappo:
                        actions_list, values_list, log_probs_list = action_output
                        actions = actions_list
                        extra = (actions_list, values_list, log_probs_list)
                    elif is_ppo:
                        actions_list, value, log_prob = action_output
                        actions = actions_list
                        extra = (actions_list, value, log_prob)
                else:
                    actions = action_output

                next_states, rewards, dones, info = env.step(actions)

                if is_on_policy:
                    if is_ppo:
                        # PPO 仅过滤 agent_id 那个 agent 的 post-done
                        if not prev_dones[model.agent_id]:
                            _push_on_policy_data(model, (states, actions, rewards, extra), next_states, dones)
                    else:
                        # MAPPO：把 prev_dones 传下去，per-agent 过滤
                        _push_on_policy_data(model, (states, actions, rewards, extra), next_states, dones,
                                             prev_dones=prev_dones)
                else:
                    _push_and_maybe_update_off_policy(
                        model, (states, actions, rewards, next_states, dones),
                        global_step + 1, train_interval,
                    )

                global_step += 1
                _sync_target(model, global_step)

                # on-policy 更新：每 update_interval 步
                if is_on_policy and global_step % update_interval == 0:
                    _update_on_policy(model, next_states, dones)

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

            # episode 结束时，如果 buffer 里还有数据，就更新一次，然后清空
            if is_on_policy:
                if is_mappo:
                    has_data = any(len(model.buffers[i]) > 0 for i in range(model.num_agents))
                    if has_data:
                        _update_on_policy(model, next_states, dones)
                    # 不管有没有更新，都清空 buffer，确保下一个 episode 从头开始
                    for i in range(model.num_agents):
                        model.buffers[i].clear()
                elif is_ppo:
                    if len(model.buffer) > 0:
                        _update_on_policy(model, next_states, dones)
                    model.buffer.clear()

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
                # 每 agent 在该 ep 的最佳信号质量：max(-log10 BER)
                max_neglog_i = float(np.max(ep_neglog_ber_per_agent[i])) if ep_neglog_ber_per_agent[i] else 0.0
                agent_ber_max_lists[i].append(max_neglog_i)
                ep_neglog_all.extend(ep_neglog_ber_per_agent[i])
            ber_list.append(float(np.mean(ep_neglog_all)) if ep_neglog_all else 0.0)
            # 全局最佳：所有 agent 所有 step 中最小 BER 对应的 -log10
            ber_max_list.append(float(np.max(ep_neglog_all)) if ep_neglog_all else 0.0)

            ep_dt = time.perf_counter() - ep_t0
            time_list.append(float(ep_dt))

            # 到达目标的 agent 数（done_flags 在 env.step 中仅在到达目标时置 True）
            reached = int(np.sum(env.done_flags))
            reached_list.append(reached)

            # 探索强度衰减
            #   off-policy (DQN/MADQN/QMIX/VDN): epsilon 衰减
            #   on-policy  (PPO/MAPPO):           entropy_coef 衰减
            # 两者作用一致：从"广泛探索"过渡到"接近确定"
            if not is_on_policy:
                model.epsilon = max(model.epsilon_min, model.epsilon * model.epsilon_decay)
            else:
                model.entropy_coef = max(model.entropy_coef_min,
                                         model.entropy_coef * model.entropy_coef_decay)

            if logger:
                explore_log = (f" eps={model.epsilon:.3f}" if not is_on_policy
                               else f" ent={model.entropy_coef:.4f}")
                logger.debug("iter %s ep %s/%s R=%.1f reached=%d/%d step=%.1f approach=%.1f comm=%.1f -log10BER=%.2f t=%.2fs%s",
                             it, ep, num_episodes, ep_total.sum(), reached, num_agents,
                             ep_step.sum(), ep_approach.sum(), ep_comm.sum(),
                             ber_list[-1], ep_dt, explore_log)

            # 第一行：核心进度（R / reached / ep）
            pbar_postfix = {
                "R": f"{ep_total.sum():.1f}",
                "reached": f"{reached}/{num_agents}",
                "ep": f"{ep_counter}/{total_eps}",
            }
            pbar.set_postfix(pbar_postfix)

            # 第二行：扩展明细（appr / comm / -logBER / t / eps 或 ent）
            explore_str = (f" eps={model.epsilon:.3f}" if not is_on_policy
                           else f" ent={model.entropy_coef:.4f}")
            detail_bar.set_description_str(
                f"  appr={ep_approach.sum():.1f}  comm={ep_comm.sum():.1f}  "
                f"-logBER={ber_list[-1]:.1f}  t={ep_dt:.1f}s{explore_str}"
            )
        detail_bar.close()

    if return_list and logger:
        avg = sum(return_list) / len(return_list)
        last10 = sum(return_list[-10:]) / min(10, len(return_list))
        logger.info("%s training done: total_eps=%d avg_return=%.2f last10_avg_return=%.2f",
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
        "ber_max_list": ber_max_list,
        "agent_ber_max_lists": agent_ber_max_lists,
        "time_list": time_list,
        "reached_list": reached_list,
    }