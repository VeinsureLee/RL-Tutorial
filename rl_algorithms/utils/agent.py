import numpy as np
import random
from typing import List, Tuple, Any, Optional


class Agent:
    """
    Agent基类模板，使用随机策略选择动作
    其他强化学习算法可以继承此类并重写相关方法
    """
    
    def __init__(self, env,
                 lr = 0.001, gamma = 0.99, 
                 epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.995,
                 num_episodes = 10, episode_length = 400000):
        # 初始化环境
        self.env = env
        
        # 初始化超参数
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        
        self.lr = lr
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, state, agent_id = None, training = True):
        """
        使用随机策略选择动作
        :param state: 状态
        :param training: 是否处于训练模式
        :return: action value估计
        """
        target = self.env.target_states[agent_id]
        if tuple(state) == tuple(target):
            return (0, 0)
        else:
            return random.choice(self.env.action_space)
    
    def train(self, *args, **kwargs):
        """
        训练方法（占位符）
        子类可以重写此方法实现具体的训练逻辑
        """
        pass
    
    def save(self, path: str):
        """
        保存模型（占位符）
        子类可以重写此方法实现模型保存逻辑
        :param path: 保存路径
        """
        pass
    
    def load(self, path: str):
        """
        加载模型（占位符）
        子类可以重写此方法实现模型加载逻辑
        :param path: 加载路径
        """
        pass
