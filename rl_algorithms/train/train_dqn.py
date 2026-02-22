"""DQN 训练逻辑。"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tqdm import tqdm
from env.env import Env
from rl_algorithms.structure.dqn import DQN


def train_dqn(env, dqn):
    """训练 DQN，返回 (dqn, return_list)。"""
    epsilon = dqn.epsilon
    return_list = []
    print(f"Begin to train DQN, iteration: {dqn.iteration}")
    for i in range(dqn.iteration):
        dqn.epsilon = epsilon
        pbar = tqdm(range(1, dqn.num_episodes + 1), desc=f"Iteration({i+1}) progress", unit="episode")
        for ep in pbar:
            states, _ = env.reset()
            state = states[dqn.agent_id] if isinstance(states, list) else states
            ep_return = 0
            for t in range(dqn.episode_length):
                action_idx = dqn.take_action(state, training=True)
                action = env.action_space[action_idx]
                step_arg = action if env.num_agents == 1 else [action] * env.num_agents
                next_state, reward, done, _ = env.step(step_arg)
                if env.num_agents > 1:
                    next_state = next_state[dqn.agent_id]
                    reward = reward[dqn.agent_id]
                    done = done[dqn.agent_id]
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
            pbar.set_postfix({'Episode': ep, 'Return': f'{ep_return:.2f}', 'Epsilon': f'{dqn.epsilon:.3f}'})
    return dqn, return_list
