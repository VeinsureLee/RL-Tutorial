import numpy as np


class NOMA:
    """
    对应论文第 2.3 节：NOMA + SIC
    """

    def __init__(self, noise_power):
        self.noise_power = noise_power

    @staticmethod
    def sort_users_by_channel(self, channel_gains):
        """
        根据信道增益排序（SIC 解码顺序）

        |h_1|^2 ≥ |h_2|^2 ≥ ...
        """
        return np.argsort(channel_gains)[::-1]

    def sinr(self, powers, channel_gains):
        """
        计算每个用户的 SINR（含 SIC）

        对应论文 SINR 公式
        """
        num_users = len(powers)
        sinr = np.zeros(num_users)

        order = self.sort_users_by_channel(channel_gains)

        for i, k in enumerate(order):
            interference = 0.0
            for j in range(i):
                interference += powers[order[j]] * channel_gains[k]

            signal = powers[k] * channel_gains[k]
            sinr[k] = signal / (interference + self.noise_power)

        return sinr

    @staticmethod
    def achievable_rate(self, sinr):
        """
        可达速率
        R_k = log2(1 + SINR_k)
        """
        return np.log2(1 + sinr)
