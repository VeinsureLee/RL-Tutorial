"""DQN 训练逻辑（对齐论文）。"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from tqdm import tqdm


def train_dqn(env, dqn, num_episodes=200, episode_length=5000, batch_size=128):
    """训练 DQN，返回 (dqn, return_list)。"""
    dqn.batch_size = batch_size
    return_list = []
    global_step = 0

    pbar = tqdm(range(1, num_episodes + 1), desc="DQN Training", unit="episode")
    for ep in pbar:
        states = env.reset()
        state = states[dqn.agent_id]
        ep_return = 0.0

        for step in range(episode_length):
            action = dqn.take_action(state, training=True)

            # 为所有 agent 生成动作（DQN 只控制 agent_id，其他随机）
            all_actions = [np.random.randint(env.n_actions) for _ in range(env.num_agents)]
            all_actions[dqn.agent_id] = action

            next_states, rewards, dones, info = env.step(all_actions)
            next_state = next_states[dqn.agent_id]
            reward = rewards[dqn.agent_id]
            done = dones[dqn.agent_id]

            ep_return += reward
            dqn.buffer.add(state, action, reward, next_state, done)

            if len(dqn.buffer) >= batch_size:
                batch = dqn.buffer.sample(batch_size)
                dqn.update(batch)

            global_step += 1
            if global_step % dqn.update_freq == 0:
                dqn.update_target_qnet()

            state = next_state
            if done:
                break

        return_list.append(ep_return)
        dqn.epsilon = max(dqn.epsilon_min, dqn.epsilon * dqn.epsilon_decay)
        pbar.set_postfix({'R': f'{ep_return:.1f}', 'eps': f'{dqn.epsilon:.3f}'})

    return dqn, return_list
