from config.param_arguments import parser
import math
import numpy as np


def watt2dbm(watt):
    """Convert power from watts to dBm."""
    if watt <= 0:
        raise ValueError("Power in watts must be greater than 0.")
    return 10 * np.log10(watt * 1000)

def dbm2watt(dbm):
    """Convert power from dBm to watts."""
    return 10 ** (dbm / 10) / 1000

def judge_los_nlos(x_list, y_list, map):
    # TODO: Judge LOS or NLOS based on map data
    model_list = ["LOS", "NLOS"]
    return

def distances_calculation(points):
    antenna_position = parser.parse_args().antenna_position
    return np.linalg.norm(points - antenna_position, axis=1)

if __name__ == "__main__":
    points = np.random.random((100, 2))
    print(points)
    print(distances_calculation(points))