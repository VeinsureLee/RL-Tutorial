
'''
Visualize the forbidden areas on the map and positions of agents
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from config.yml_config import _get_env_parser

args = _get_env_parser().parse_args()
map_size = tuple(int(x) for x in args.map_size) if hasattr(args.map_size, '__iter__') else args.map_size
forbidden_areas = args.forbidden_areas
start_states = args.start_states
target_states = args.target_state

# visualize the map with forbidden areas and agents
def visualize_map_with_agents(map_size, forbidden_areas, start_states, target_states):
    """
    Visualize the map with forbidden areas and agents' start/target positions
    :param map_size: the size of the map (tuple, e.g. (rows, cols))
    :param forbidden_areas: a list of forbidden area definitions ((x, y), size)
    :param start_states: list of agent start positions [(x1, y1), ...]
    :param target_states: list of agent target positions [(x1, y1), ...]
    """
    # Create a base map array
    map_array = np.ones(map_size)

    # Mark forbidden areas
    for square_position, square_size in forbidden_areas:
        x, y = square_position
        map_array[x:x + square_size, y:y + square_size] = 0

    plt.figure(figsize=(6, 6))
    plt.imshow(map_array, cmap='gray', origin='lower')
    plt.colorbar(label="Forbidden Area (0 = forbidden)")
    plt.title('Map with Forbidden Areas and Agent States')
    plt.xlabel('Y coordinate')
    plt.ylabel('X coordinate')

    # Plot start states
    if start_states:
        start_states = np.array(start_states)
        plt.scatter(start_states[:,1], start_states[:,0], c='g', marker='o', s=100, label='Start', edgecolors='black')
        # Add labels for each start state
        for i, (x, y) in enumerate(start_states):
            plt.text(y, x, f'start{i+1}', fontsize=8, ha='left', va='bottom', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Plot target states
    if target_states:
        target_states = np.array(target_states)
        plt.scatter(target_states[:,1], target_states[:,0], c='r', marker='*', s=180, label='Target', edgecolors='black')
        # Add labels for each target state
        for i, (x, y) in enumerate(target_states):
            plt.text(y, x, f'target{i+1}', fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    plt.legend()
    plt.show()

# 可视化
visualize_map_with_agents(map_size, forbidden_areas, start_states, target_states)
print(start_states)
print(target_states)
