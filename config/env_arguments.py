
'''
Specify parameters of the environment
'''
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union
import numpy as np
import argparse
from config.map_config import map_size, forbidden_areas, start_states, target_states


env_parser = argparse.ArgumentParser("Settings for Environment")


## ==================== User settings ====================
# specify the size of the environment
env_parser.add_argument("--map-size", type=Union[list, tuple, np.ndarray],
                        default=map_size, )

# specify the size of the grid
env_parser.add_argument("--grid_size", type=float, default=0.4)

# specify the start state
env_parser.add_argument("--start-states", type=Union[list, tuple, np.ndarray],
                        default=start_states)

# specify the target state
env_parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray],
                        default=target_states)

# specify the forbidden states
env_parser.add_argument("--forbidden-areas", type=eval,
                        default=forbidden_areas)

# specify the reward when reaching target
env_parser.add_argument("--reward-target", type=float,
                        default=10)

# specify the reward when entering into forbidden area
env_parser.add_argument("--reward-forbidden", type=float,
                        default=-5)

# specify the reward for each step
env_parser.add_argument("--reward-step", type=float,
                        default=-1)

# specify the reward for each step closer to target
env_parser.add_argument("--reward-closer-to-target", type=float,
                        default=1)
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
env_parser.add_argument("--action-space", type=Union[list, tuple, np.ndarray],
                        default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)])  # down, right, up, left, stay
env_parser.add_argument("--debug", type=bool,
                        default=False)
env_parser.add_argument("--animation-interval", type=float,
                        default=0.2)
## ==================== End of Advanced settings ====================


args = env_parser.parse_args()


# 验证函数由 config.utils 提供，此处复用
from config.utils import validate_environment_parameters


try:
    validate_environment_parameters(args.map_size, args.start_states, args.target_state, args.forbidden_areas)
except (ValueError, AssertionError) as e:
    print("Error:", e)