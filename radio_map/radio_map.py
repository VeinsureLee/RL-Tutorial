import numpy as np
class RadioMap:
    """
    Radio Map 实现
    对应论文中的无线电地图 M(x, y)
    """

    def __init__(self,
                 area_size=(100, 100),
                 grid_size=5,
                 num_samples=100,
                 forbidden_areas=None
                 ):
        """
        area_size : 区域大小 (m, m)
        grid_size : 栅格边长 (m)
        num_samples : 小尺度衰落采样次数（期望）
        """
        self.area_size = area_size
        self.grid_size = grid_size
        self.num_samples = num_samples

        self.x_grid = np.arange(0, area_size[0], grid_size)
        self.y_grid = np.arange(0, area_size[1], grid_size)

        self.forbidden_areas = forbidden_areas

        self.map = None

    def build_channel_gain_map(self, channel, bs_position=(0, 0)):
        """
        构建信道增益 Radio Map
        """
        gain_map = np.zeros((len(self.x_grid), len(self.y_grid)))

        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                distance = np.sqrt(
                    (x - bs_position[0]) ** 2 +
                    (y - bs_position[1]) ** 2
                ) + 1e-3

                samples = []
                for _ in range(self.num_samples):
                    samples.append(channel.channel_gain(distance))

                gain_map[i, j] = np.mean(samples)

        self.map = gain_map
        return gain_map

    def build_sinr_map(self,
                       channel,
                       tx_power=1.0,
                       bs_position=(0, 0)):
        """
        构建 SINR Radio Map
        """
        sinr_map = np.zeros((len(self.x_grid), len(self.y_grid)))

        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                distance = np.sqrt(
                    (x - bs_position[0]) ** 2 +
                    (y - bs_position[1]) ** 2
                ) + 1e-3

                samples = []
                for _ in range(self.num_samples):
                    h2 = channel.channel_gain(distance)
                    samples.append(tx_power * h2 / channel.noise_power)

                sinr_map[i, j] = np.mean(samples)

        return sinr_map

    @staticmethod
    def build_rate_map(self, sinr_map):
        """
        构建速率 Radio Map
        """
        return np.log2(1 + sinr_map)
