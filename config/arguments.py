__credits__ = ["Graduation project at BUPT."]
'''
Specify parameters of the environment
'''
from typing import Union
import numpy as np
import argparse

parser = argparse.ArgumentParser("Map settings for Environment")

## ==================== Forbidden area settings ====================
forbidden_states = []
# 添加 (i,1)，i从0到4（range(5) 是 0,1,2,3,4）
forbidden_states.extend([(i, 1) for i in range(5)])
# 添加 (1,j)，j从2到5（range(2,6) 是 2,3,4,5）
forbidden_states.extend([(1, j) for j in range(2, 6)])
# 添加 (i,3)，i从3到5（range(3,6) 是 3,4,5）
forbidden_states.extend([(i, 3) for i in range(3, 5)])
# 添加单独的禁止状态
forbidden_states.extend([(5, 4), (3, 5)])
## ==================== Forbidden area settings ====================


## ==================== User settings ===================='''
# specify the frequency of the carrier in GHz
parser.add_argument("--carrier-frequency", type=float, default=3.5)

# specify the sigma of rayleigh distribution (Usually set 1.2 in door)
parser.add_argument("--sigma_rayleigh", type=float, default=1.2)

# specify the number of antenna
parser.add_argument("--number_of_antenna", type=int, default=128)

# specify the antenna position in map
parser.add_argument("--antenna_position", type=Union[list, tuple, np.ndarray], default=(24,12))

# specify the power of AWGN in dbm/Hz
parser.add_argument("--power_AWGN", type=float, default=-143.0)

# specify the channel block length
parser.add_argument("--channel_block_length", type=int, default=256)

# specify the packet size
parser.add_argument("--packet_size", type=int, default=16)

# specify the number of robots
parser.add_argument("--number_of_robots", type=int, default=4)

# specify the size of the environment
parser.add_argument("--map-size", type=Union[list, tuple, np.ndarray], default=(48, 24))

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