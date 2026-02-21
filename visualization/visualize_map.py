
'''
Visualize the forbidden areas on the map
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from config.yml_config import get_map_and_scenario

_map_size, _forbidden_areas, _, _, _ = get_map_and_scenario()
map_size = tuple(int(x) for x in _map_size)
forbidden_areas = _forbidden_areas


# visualize the forbidden areas by matplotlib
def visualize_forbidden_areas(map_size, forbidden_areas):
    """
    Visualize the forbidden areas on the map by matplotlib
    :param map_size: the size of map
    :param forbidden_areas: the forbidden areas
    """
    
    # 先创建地图数组（2D数组）
    map_array = np.ones(map_size)
    for square_position, square_size in forbidden_areas:
        x, y = square_position
        map_array[x:x + square_size, y:y + square_size] = 0
    
    # 然后显示地图数组
    plt.figure(figsize=(6, 6))  # 修正：figsize 应该是 (width, height) 元组，而不是 map_size
    plt.imshow(map_array, cmap='gray', origin='lower')  # origin='lower' 使坐标从底部开始
    plt.colorbar(label="Forbidden Area (0 = forbidden)")
    plt.title('Forbidden Areas on Map')
    plt.xlabel('Y coordinate')
    plt.ylabel('X coordinate')
    plt.show()
    
    return map_array

print(map_size)

print(forbidden_areas)

visualize_forbidden_areas(map_size, forbidden_areas)
