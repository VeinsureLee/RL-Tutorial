
'''
Specify parameters of the environment
'''
from typing import Union
import numpy as np
import argparse
from .map_config import *


env_parser = argparse.ArgumentParser("Settings for Environment")


## ==================== User settings ====================
# specify the size of the environment
env_parser.add_argument("--map-size", type=Union[list, tuple, np.ndarray], default=map_size, )

# specify the size of the grid
env_parser.add_argument("--grid_size", type=float, default=grid_size)

# specify the start state
env_parser.add_argument("--start-states", type=Union[list, tuple, np.ndarray], default=start_states)

# specify the target state
env_parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=target_state)

# specify the forbidden states
env_parser.add_argument("--forbidden-states", type=eval, default=forbidden_states)

# specify the reward when reaching target
env_parser.add_argument("--reward-target", type=float, default=reward_target)

# specify the reward when entering into forbidden area
env_parser.add_argument("--reward-forbidden", type=float, default=reward_forbidden)

# specify the reward for each step
env_parser.add_argument("--reward-step", type=float, default=reward_step)
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
env_parser.add_argument("--action-space", type=Union[list, tuple, np.ndarray],
                        default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)])  # down, right, up, left, stay
env_parser.add_argument("--debug", type=bool, default=False)
env_parser.add_argument("--animation-interval", type=float, default=0.2)
## ==================== End of Advanced settings ====================


args = env_parser.parse_args()


def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) and len(
            env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")

    for i in range(2):
        assert start_state[i] < env_size[i]
        assert target_state[i] < env_size[i]
        for j in range(len(forbidden_states)):
            assert forbidden_states[j][i] < env_size[i]


try:
    validate_environment_parameters(args.map_size, args.start_states, args.target_state, args.forbidden_states)
except ValueError as e:
    print("Error:", e)