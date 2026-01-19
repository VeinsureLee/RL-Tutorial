
"""
This file combines map settings and forbidden area generation
You can configure map parameters and random square forbidden areas
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union
import numpy as np
from config.param_arguments import parser


antenna_position = parser.parse_args().antenna_position


# ==================== Map settings ====================
# specify the size of map
map_size: Union[list, tuple, np.ndarray] = (48, 24)
# ==================== End of map settings ====================


# ==================== Forbidden areas basic settings ====================
# specify the random seed
RANDOM_SEED = 42

# specify the number of forbidden areas(squares)
NUM_FORBIDDEN_SQUARES = 5

# specify the range of square size
SQUARE_SIZE_RANGE = (3, 5)
# ==================== End of forbidden areas basic settings ====================


# ==================== Logistic of forbidden areas generation ====================
def generate_forbidden_areas(map_size, 
                             antenna_position, 
                             num_forbidden_squares, 
                             square_size_range, 
                             random_seed):
    """
        Generate forbidden areas on the map, 
        forbidden areas can't cover the antenna position, 
        and can't overlap with each other
        :param map_size: the size of map
        :param antenna_position: the position of antenna
        :param num_forbidden_squares: the number of forbidden areas(squares)
        :param square_size_range: the range of square size
        :param random_seed: the random seed
        :return: the forbidden areas
    """
    np.random.seed(random_seed)
    forbidden_areas = []
    for _ in range(num_forbidden_squares):
        square_size = np.random.randint(square_size_range[0], square_size_range[1])
        square_position = (np.random.randint(0, map_size[0] - square_size), np.random.randint(0, map_size[1] - square_size))
        while square_position in forbidden_areas or square_position in antenna_position:
            square_size = np.random.randint(square_size_range[0], square_size_range[1])
            square_position = (np.random.randint(0, map_size[0] - square_size), np.random.randint(0, map_size[1] - square_size))
        forbidden_areas.append((square_position, square_size))
    return forbidden_areas
# ==================== End of logistic of forbidden areas generation ====================

forbidden_areas = generate_forbidden_areas(map_size, antenna_position,
                                           NUM_FORBIDDEN_SQUARES, 
                                           SQUARE_SIZE_RANGE,
                                           RANDOM_SEED)

# specify the variables to be exported
__all__ = [
    "map_size", "forbidden_areas",
    "RANDOM_SEED", "SQUARE_SIZE_RANGE", "NUM_FORBIDDEN_SQUARES"
]
