import numpy as np


class Channel:
    """
    对应论文第 2.2 节：信道模型
    """

    def __init__(self,
                 noise_power_dbm=-174,
                 bandwidth=1e6):
        self.bandwidth = bandwidth
        self.noise_power = self.dbm_to_watt(noise_power_dbm) * bandwidth

    @staticmethod
    def dbm_to_watt(dbm):
        """
        噪声功率单位转换
        """
        return 10 ** ((dbm - 30) / 10)

    def path_loss(self, distances, LOS=True):
        """
        大尺度路径损耗（工业环境可调）
        """
        pl_los = 31.87 + 21.5 * np.log10(distances) + 19 * np.log10(self.fc)
        if LOS:
            pl_db = pl_los
        else:
            pl_sl = 33 + 25.5 * np.log10(distances) + 20 * np.log10(self.fc)
            pl_db = max(pl_sl, pl_los)

        return self.dbm_to_watt(pl_db)

    @staticmethod
    def small_scale_fading(self):
        """
        瑞利衰落
        g_k ~ CN(0,1)

        对应论文小尺度衰落模型
        """
        real = np.random.randn()
        imag = np.random.randn()
        return (real + 1j * imag) / np.sqrt(2)

    def channel_gain(self, distance):
        """
        总信道增益 |h_k|^2

        h_k = g_k * sqrt(L(d_k))
        """
        g = self.small_scale_fading()
        L = self.path_loss(distance)
        return np.abs(g) ** 2 * L

