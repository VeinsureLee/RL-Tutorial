__credits__ = ["Graduation project at BUPT."]
'''
Specify parameters of the environment
'''
from typing import Union
import numpy as np
import argparse


parser = argparse.ArgumentParser("Settings for Environment")


## ==================== User settings ====================
# specify the size of the environment
parser.add_argument("--map-grid-size", type=Union[list, tuple, np.ndarray], default=(48, 24))

# specify the size of the grid
parser.add_argument("--grid_size", type=float, default=0.4)

# specify the start state
parser.add_argument("--start-states", type=Union[list, tuple, np.ndarray], default=(0, 0))

# specify the target state
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(4, 4))

# specify the forbidden states
parser.add_argument("--forbidden-states", type=eval, default=forbidden_states)

# specify the reward when reaching target
parser.add_argument("--reward-target", type=float, default=10)

# specify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default=-5)

# specify the reward for each step
parser.add_argument("--reward-step", type=float, default=-1)
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
parser.add_argument("--action-space", type=Union[list, tuple, np.ndarray],
                    default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)])  # down, right, up, left, stay
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--animation-interval", type=float, default=0.2)
## ==================== End of Advanced settings ====================


args = parser.parse_args()


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