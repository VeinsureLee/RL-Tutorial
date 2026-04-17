"""DQN 训练逻辑：外层 iteration + 内层 episode，每轮 iteration 重置 epsilon。"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from tqdm import tqdm


def train_dqn(env, dqn, num_iterations=10, num_episodes=200, episode_length=5000,
              batch_size=128, train_interval=4):
    """
    训练 DQN，返回 (dqn, return_list)。
    外层 iteration 循环：每轮重置 epsilon 到初始值再衰减，避免局部最优。
    """
    dqn.batch_size = batch_size
    return_list = []
    global_step = 0
    epsilon_start = dqn.epsilon  # 保存初始 epsilon

    total_episodes = num_iterations * num_episodes
    ep_count = 0

    for it in range(1, num_iterations + 1):
        # 每轮 iteration 重置 epsilon
        dqn.epsilon = epsilon_start

        pbar = tqdm(range(1, num_episodes + 1),
                    desc=f"DQN Iter {it}/{num_iterations}",
                    unit="ep")
        for ep in pbar:
            ep_count += 1
            states = env.reset()
            state = states[dqn.agent_id]
            ep_return = 0.0

            for step in range(episode_length):
                action = dqn.take_action(state, training=True)

                all_actions = [np.random.randint(env.n_actions) for _ in range(env.num_agents)]
                all_actions[dqn.agent_id] = action

                next_states, rewards, dones, info = env.step(all_actions)
                next_state = next_states[dqn.agent_id]
                reward = rewards[dqn.agent_id]
                done = dones[dqn.agent_id]

                ep_return += reward
                dqn.buffer.add(state, action, reward, next_state, done)

                global_step += 1
                if global_step % train_interval == 0:
                    if len(dqn.buffer) >= batch_size:
                        batch = dqn.buffer.sample(batch_size)
                        dqn.update(batch)
                if global_step % dqn.update_freq == 0:
                    dqn.update_target_qnet()

                state = next_state
                if done:
                    break

            return_list.append(ep_return)
            dqn.epsilon = max(dqn.epsilon_min, dqn.epsilon * dqn.epsilon_decay)
            pbar.set_postfix({
                'R': f'{ep_return:.1f}',
                'eps': f'{dqn.epsilon:.3f}',
                'total': f'{ep_count}/{total_episodes}',
            })

    return dqn, return_list
