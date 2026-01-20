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
                # 若agent已到达target，则始终停留在原地
                actions = []
                for i, s in enumerate(state):
                    # 获取对应target
                    target = self.env.target_states[i] if hasattr(self.env, "target_states") else None
                    if target is not None and tuple(s) == tuple(target):
                        # 停留在原地（假设action_space中有(0,0))
                        stay_action = (0, 0)
                        # 如果(0,0)不在action_space，则选择第一个动作
                        if stay_action in self.action_space:
                            actions.append(stay_action)
                        else:
                            actions.append(self.action_space[0])
                    else:
                        actions.append(random.choice(self.action_space))
                return actions
            else:
                # 单个agent的情况，但环境支持多agent
                # 也检查state是否已到target
                target = self.env.target_states[0] if hasattr(self.env, "target_states") else None
                if target is not None and tuple(state) == tuple(target):
                    stay_action = (0, 0)
                    if stay_action in self.action_space:
                        return stay_action
                    else:
                        return self.action_space[0]
                return random.choice(self.action_space)
        else:
            # 单agent环境
            target = self.env.target_state if hasattr(self.env, "target_state") else None
            if target is not None and tuple(state) == tuple(target):
                stay_action = (0, 0)
                if stay_action in self.action_space:
                    return stay_action
                else:
                    return self.action_space[0]
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
