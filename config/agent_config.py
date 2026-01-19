
'''
Specify parameters of the environment
'''
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union
import numpy as np
import argparse
from config.param_arguments import parser
from config.map_config import *


num_agents = parser.parse_args().number_of_robots


# ========== Generate start states and target states ==========
def generate_states(num_agents, forbidden_areas):
    
    np.random.seed(RANDOM_SEED)
    start_states = []
    target_states = []

    occupied_positions = set()
    for forbidden in forbidden_areas:
        # forbidden: ((row, col), size)
        pos, size = forbidden
        for r in range(pos[0], pos[0] + size):
            for c in range(pos[1], pos[1] + size):
                occupied_positions.add((r, c))

    def is_valid(pos):
        return (pos not in occupied_positions and
                0 <= pos[0] < map_size[0] and
                0 <= pos[1] < map_size[1])

    # Generate unique valid start and target states for each agent
    while len(start_states) < num_agents:
        candidate = (np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1]))
        if is_valid(candidate) and candidate not in start_states:
            start_states.append(candidate)
            occupied_positions.add(candidate)  # avoid overlapping start/target

    while len(target_states) < num_agents:
        candidate = (np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1]))
        if is_valid(candidate) and candidate not in target_states and candidate not in start_states:
            target_states.append(candidate)
            occupied_positions.add(candidate)

    return start_states, target_states
# ========== End of generation ==========

start_states, target_states = generate_states(num_agents=num_agents,
                                              forbidden_areas=forbidden_areas)

__all__ = [
    "start_states", "target_states"
]
