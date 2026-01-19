import numpy as np
import random
from typing import List, Tuple, Any, Optional


class Agent:
    """
    Agent基类模板，使用随机策略选择动作
    其他强化学习算法可以继承此类并重写相关方法
    """
    
    def __init__(self, env):
        """
        初始化Agent
        :param env: 环境实例，必须包含action_space属性
        """
        self.env = env
        self.action_space = env.action_space
        self.num_actions = len(env.action_space)
        
    def select_action(self, state: Any, agent_id: Optional[int] = None, training: bool = True) -> Any:
        """
        使用随机策略选择动作
        
        :param state: 当前状态（可以是单个状态或状态列表）
        :param agent_id: agent的ID（可选，用于多agent场景）
        :param training: 是否处于训练模式（可选，用于区分训练和测试）
        :return: 选择的动作（单个动作或动作列表）
        """
        # 如果是多agent环境，为每个agent随机选择动作
        if hasattr(self.env, 'num_agents') and self.env.num_agents > 1:
            # 如果state是列表，说明是多个agent的状态
            if isinstance(state, (list, tuple)) and len(state) == self.env.num_agents:
                # 为每个agent随机选择一个动作
                actions = [random.choice(self.action_space) for _ in range(self.env.num_agents)]
                return actions
            else:
                # 单个agent的情况，但环境支持多agent
                return random.choice(self.action_space)
        else:
            # 单agent环境
            return random.choice(self.action_space)
    
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
