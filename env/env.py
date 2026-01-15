import numpy as np
from config.param_arguments import parser


class Env:
    def __init__(self):
        self.size = parser.parse_args().map_size
        self.grid_size = parser.parse_args().grid_size
        self.start_states = parser.parse_args().start_states
        self.forbidden = parser.parse_args().forbidden