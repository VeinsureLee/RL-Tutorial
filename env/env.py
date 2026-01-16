import numpy as np
from config.env_arguments import env_parser


class Env:
    def __init__(self):
        self.traj = []

        self.map_size = env_parser.parse_args().map_size
        self.grid_size = env_parser.parse_args().grid_size
        self.size = (self.map_size[0] / self.grid_size, self.map_size[0] / self.grid_size)

        self.start_state = env_parser.parse_args().start_states
        self.target_state = env_parser.parse_args().target_states
        self.forbidden_states = env_parser.parse_args().forbidden_states

        self.agent_state = env_parser.parse_args().start_states
        self.num_actions = len(env_parser.parse_args().action_space)
        self.action_space = env_parser.parse_args().action_space

        self.reward_target = env_parser.parse_args().reward_target
        self.reward_forbidden = env_parser.parse_args().reward_forbidden
        self.reward_step = env_parser.parse_args().reward_step

    # Reset the environment to the start state
    def reset(self):
        self.agent_state = self.start_state
        self.traj = [self.agent_state]
        return self.agent_state, {}

    # Take a step in the environment
    def step(self, action):
        assert action in self.action_space, "Invalid action"

        next_state, reward = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store, y_store)) + 0.2 * np.array(action))
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}

    # Determine the next state and reward based on current state and action
    def _get_next_state_and_reward(self, state, action):
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))
        if y + 1 > self.size[1] - 1 and action == (0, 1):  # down
            y = self.size[1] - 1
            reward = self.reward_forbidden
        elif x + 1 > self.size[0] - 1 and action == (1, 0):  # right
            x = self.size[0] - 1
            reward = self.reward_forbidden
        elif y - 1 < 0 and action == (0, -1):  # up
            y = 0
            reward = self.reward_forbidden
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden
        elif new_state == self.target_state:  # stay
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # stay
            x, y = state
            reward = self.reward_forbidden
        else:
            x, y = new_state
            reward = self.reward_step

        return (x, y), reward

    # Check if the current state is the target state
    def _is_done(self, state):
        return state == self.target_state
