import numpy as np
from communication.channel import Channel
from communication.noma import NOMA
from communication.ofdma import OFDM


class Env:
    """
    对应论文第三章：强化学习环境
    """

    def __init__(self,
                 num_users=4,
                 access_mode="NOMA",
                 max_power=1.0):

        self.channel_gains = None
        self.distances = None
        self.num_users = num_users
        self.max_power = max_power

        self.channel = Channel()
        self.access_mode = access_mode

        if access_mode == "NOMA":
            self.access = NOMA(self.channel.noise_power)
        else:
            self.access = OFDM(self.channel.noise_power)

        self.reset()

    def reset(self):
        """
        环境初始化
        """
        self.distances = np.random.uniform(5, 50, self.num_users)
        self.channel_gains = np.array(
            [self.channel.channel_gain(d) for d in self.distances]
        )
        return self.state()

    def state(self):
        """
        状态定义（对应论文 state）
        """
        return self.channel_gains.copy()

    def step(self, action):
        """
        动作：功率分配向量
        满足 sum(P_k) ≤ P_max
        """
        powers = np.clip(action, 0, self.max_power)
        powers = powers / (np.sum(powers) + 1e-9) * self.max_power

        sinr = self.access.sinr(powers, self.channel_gains)
        rate = self.access.achievable_rate(sinr)

        reward = np.sum(rate)  # 系统总速率（论文目标）

        next_state = self.reset()
        done = False

        return next_state, reward, done, {
            "sinr": sinr,
            "rate": rate
        }
