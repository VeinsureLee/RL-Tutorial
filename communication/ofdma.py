import numpy as np


class OFDM:
    """
    对应论文第 2.4 节：OFDMA
    """

    def __init__(self, noise_power):
        self.noise_power = noise_power

    def sinr(self, powers, channel_gains):
        """
        OFDMA 下 SINR（无干扰）
        """
        return (powers * channel_gains) / self.noise_power

    @staticmethod
    def achievable_rate(self, sinr):
        """
        可达速率
        """
        return np.log2(1 + sinr)
