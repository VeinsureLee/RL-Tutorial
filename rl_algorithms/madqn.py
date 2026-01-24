import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.env import Env
from rl_algorithms.agent import Agent
from rl_algorithms.dqn import Qnet, ReplayBuffer

from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from typing import List, Tuple, Any, Optional
from collections import deque
import random
import numpy as np


class MultiQnet(nn.Module):
    """
    多Agent Q网络，包含num_agents个Qnet
    每个agent分开训练
    """

    def __init__(
        self,
        num_agents: int,
        state_num: int,
        action_dim: int,
        x_dim: int,
        y_dim: int,
        target_states: list,  # 每个agent的target_xy列表
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.num_agents = num_agents
        
        # 为每个agent创建独立的Qnet
        self.qnets = nn.ModuleList([
            Qnet(
                state_num=state_num,
                action_dim=action_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                target_xy=target_states[i],
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )
            for i in range(num_agents)
        ])
    
    def forward(self, state_idx: torch.Tensor, agent_ids: torch.Tensor):
        """
        根据agent_id选择对应的Qnet进行前向传播（优化版本，减少循环）
        state_idx: (B,)
        agent_ids: (B,) - 每个样本对应的agent id
        返回: (B, action_dim) - 每个样本的Q值
        """
        batch_size = state_idx.shape[0]
        action_dim = self.qnets[0].head[-1].out_features
        q_values = torch.zeros(batch_size, action_dim, 
                               device=state_idx.device, dtype=torch.float32)
        
        # 获取唯一的agent_id并批量处理
        unique_agent_ids = torch.unique(agent_ids)
        
        # 为每个唯一的agent_id批量计算Q值
        for agent_id in unique_agent_ids:
            mask = (agent_ids == agent_id)
            if mask.any():
                q_values[mask] = self.qnets[agent_id.item()](state_idx[mask])
        
        return q_values


class MADQN:
    def __init__(self, env,
                 lr=0.001, gamma=0.99, iteration=5,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 num_episodes=10, episode_length=400000,
                 batch_size=64, mini_batch_size=32,
                 hidden_dim=128, update_freq=10):
        # 不调用super().__init__，因为我们需要为每个agent创建独立的qnet
        # 只初始化必要的父类属性
        from rl_algorithms.agent import Agent
        Agent.__init__(self, env=env,
                       lr=lr, gamma=gamma,
                       epsilon=epsilon, epsilon_min=epsilon_min,
                       epsilon_decay=epsilon_decay, num_episodes=num_episodes, episode_length=episode_length)

        self.iteration = iteration
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.hidden_dim = hidden_dim
        self.update_freq = update_freq

        # 为每个agent创建独立的Q网络和优化器
        self.num_agents = env.num_agents
        state_num = env.state_num
        action_dim = env.num_actions
        x_dim = env.x_dim
        y_dim = env.y_dim

        # 初始化设备
        self.device = torch.device("cpu")

        # 创建MultiQnet（包含所有agent的Qnet）
        self.qnet = MultiQnet(
            num_agents=self.num_agents,
            state_num=state_num,
            action_dim=action_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            target_states=env.target_states,
            embedding_dim=64,
            hidden_dim=hidden_dim,
        )
        
        # 创建target MultiQnet
        self.target_qnet = MultiQnet(
            num_agents=self.num_agents,
            state_num=state_num,
            action_dim=action_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            target_states=env.target_states,
            embedding_dim=64,
            hidden_dim=hidden_dim,
        )
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        self.qnet.to(self.device)
        self.target_qnet.to(self.device)

        # 创建优化器（优化所有Qnet参数）
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # 每个agent独立的epsilon
        self.epsilons = [epsilon] * self.num_agents

        # 为每个agent创建独立的经验回放缓冲区
        buffer_size = episode_length * num_episodes
        self.buffers = [
            ReplayBuffer(max_size=buffer_size)
            for _ in range(self.num_agents)
        ]
        self.loss_fn = nn.MSELoss()

    def _state_to_idx_tensor(self, state):
        """
        将状态转换为 state idx 张量
        :param state: 单个状态 (x, y) 或批量状态数组 (batch, 2)
        :return: state idx tensor，单个状态返回形状 (1,)，批量状态返回形状 (batch,)
        """
        if isinstance(state, (tuple, list, np.ndarray)) and len(state) == 2 and isinstance(state[0], (int, np.integer)):
            # 单个状态 (x, y)
            state_idx = int(state[0]) * self.env.y_dim + int(state[1])
            state_idx_tensor = torch.tensor(
                [state_idx], dtype=torch.long, device=self.device)
        else:
            # 批量状态 (batch, 2)
            state = np.array(state, dtype=np.int64)
            state_indices = state[:, 0] * self.env.y_dim + state[:, 1]
            state_idx_tensor = torch.tensor(
                state_indices, dtype=torch.long, device=self.device)
        return state_idx_tensor

    def take_action(self, states, agent_ids, training=True):
        """
        选择动作（多agent版本）
        :param states: 状态列表或单个状态，如果是列表则对应多个agent
        :param agent_ids: agent_id列表或单个agent_id
        :param training: 是否处于训练模式
        :return: action索引列表或单个action索引
        """
        # 保存原始输入类型
        is_single_state = not isinstance(states, list)
        is_single_agent_id = not isinstance(agent_ids, list)
        
        # 处理单个状态的情况
        if is_single_state:
            states = [states]
        if is_single_agent_id:
            agent_ids = [agent_ids]
        
        # 确保states和agent_ids长度一致
        assert len(states) == len(agent_ids), "states和agent_ids长度必须一致"
        
        actions = []
        for i, (state, agent_id) in enumerate(zip(states, agent_ids)):
            target = self.env.target_states[agent_id]
            if state == target:
                actions.append(0)  # 默认返回第0个动作
            else:
                # 将 (x, y) 转换为 state_idx: state_idx = x * y_dim + y
                state_idx_tensor = self._state_to_idx_tensor(state)
                agent_id_tensor = torch.tensor([agent_id], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    action_values = self.qnet(state_idx_tensor, agent_id_tensor)

                # 获取最优动作
                optimal_action = action_values.argmax(dim=1).item()

                if training:
                    # 概率分配：最优动作概率最大，其他动作概率相同
                    num_actions = self.env.num_actions
                    epsilon = self.epsilons[agent_id]
                    probs = torch.full((num_actions,), epsilon /
                                       num_actions, device=self.device)
                    probs[optimal_action] = 1 - epsilon + epsilon / num_actions

                    # 根据概率分布采样动作
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample().item()
                    actions.append(action)
                else:
                    # 测试模式：直接返回最优动作
                    actions.append(optimal_action)
        
        # 如果输入是单个值，返回单个值；否则返回列表
        if is_single_state and is_single_agent_id and len(actions) == 1:
            return actions[0]
        return actions

    def update(self):
        """
        更新所有agent的Q网络
        从每个agent的独立buffer中采样，并按照multi qnet计算loss
        每个agent的loss根据agent id选取对应的td target进行更新
        每个agent分开训练，直接求和loss
        """
        # 从每个agent的独立buffer中采样
        loss_components = []
        for agent_id in range(self.num_agents):
            # 如果该agent的buffer样本不足，跳过
            if len(self.buffers[agent_id]) < self.mini_batch_size:
                continue
            
            # 从该agent的buffer中采样
            batch = self.buffers[agent_id].sample(self.mini_batch_size)
            
            # 解析batch数据
            states = np.array([b[1] for b in batch])
            actions = np.array([b[2] for b in batch])
            rewards = np.array([b[3] for b in batch], dtype=np.float32)
            next_states = np.array([b[4] for b in batch])
            dones = np.array([b[5] for b in batch], dtype=np.float32)
            
            # 转换为tensor
            action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            reward_tensor = torch.tensor(rewards, device=self.device)
            done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
            
            # 转换为state_idx
            agent_state_idx = self._state_to_idx_tensor(states)
            agent_next_state_idx = self._state_to_idx_tensor(next_states)
            
            # 使用该agent对应的qnet计算Q值
            agent_q = self.qnet.qnets[agent_id](agent_state_idx)
            agent_q_selected = agent_q.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            
            # 使用该agent对应的target_qnet计算td_target
            with torch.no_grad():
                agent_next_q = self.target_qnet.qnets[agent_id](agent_next_state_idx)
                agent_max_next_q = agent_next_q.max(dim=1)[0]
                agent_td_target = reward_tensor + self.gamma * agent_max_next_q * (1 - done_tensor)
            
            # 计算该agent的loss
            agent_loss = self.loss_fn(agent_q_selected, agent_td_target)
            loss_components.append(agent_loss)
        
        # 如果没有任何agent的样本，返回0
        if len(loss_components) == 0:
            return 0.0
        
        # 计算总loss（直接求和）
        total_loss = sum(loss_components)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def update_target_qnet(self):
        """同步所有agent的目标Q网络参数"""
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def save(self, path_prefix: str):
        """保存所有agent的模型"""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        path = f"{path_prefix}.pth"
        torch.save({
            "qnet": self.qnet.state_dict(),
            "target_qnet": self.target_qnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilons": self.epsilons,
        }, path)
        print(f"模型已保存到: {path}")

    def load(self, path_prefix: str):
        """加载所有agent的模型"""
        path = f"{path_prefix}.pth"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到模型文件: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(checkpoint["qnet"])
        self.target_qnet.load_state_dict(
            checkpoint.get("target_qnet", checkpoint["qnet"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilons = checkpoint.get("epsilons", self.epsilons)


def train_madqn(madqn: MADQN):
    """
    训练多个DQN agent的外部函数
    :param madqn: MADQN实例
    """
    print(
        f"开始训练多Agent DQN, iteration: {madqn.iteration}, agents: {madqn.num_agents}")

    # 保存初始epsilon值
    initial_epsilons = madqn.epsilons.copy()

    for i in range(madqn.iteration):
        # 每个iteration开始时重置epsilon
        madqn.epsilons = initial_epsilons.copy()

        # 打印iteration开始信息
        print(f"\n{'='*60}")
        print(f"开始 Iteration {i+1}/{madqn.iteration}")
        print(f"{'='*60}")

        # 使用tqdm创建进度条，每个iteration独立显示
        pbar = tqdm(range(1, madqn.num_episodes + 1),
                    desc=f"Iteration {i+1}/{madqn.iteration}",
                    unit="episode",
                    leave=True)  # leave=True 保留进度条

        for ep in pbar:
            states, _ = madqn.env.reset()
            # 确保states是列表
            if not isinstance(states, list):
                states = [states]

            ep_returns = [0.0] * madqn.num_agents
            step_count = 0

            for t in range(madqn.episode_length):
                # 为每个agent选择动作 - 使用新的take_action接口
                agent_ids = list(range(madqn.num_agents))
                action_indices = madqn.take_action(states, agent_ids, training=True)
                
                # 确保action_indices是列表
                if not isinstance(action_indices, list):
                    action_indices = [action_indices]
                
                actions = [madqn.env.action_space[action_idx] for action_idx in action_indices]

                # 执行动作
                next_states, rewards, dones, _ = madqn.env.step(actions)

                # 确保返回的是列表
                if not isinstance(next_states, list):
                    next_states = [next_states]
                if not isinstance(rewards, list):
                    rewards = [rewards]
                if not isinstance(dones, list):
                    dones = [dones]

                # 存储经验到每个agent的独立buffer
                for agent_id in range(madqn.num_agents):
                    state = states[agent_id]
                    target = madqn.env.target_states[agent_id]

                    ep_returns[agent_id] += rewards[agent_id]

                    madqn.buffers[agent_id].add(
                        agent_id,
                        state,
                        action_indices[agent_id],
                        rewards[agent_id],
                        next_states[agent_id],
                        dones[agent_id],
                        target
                    )

                # 如果所有agent的缓冲区都有足够样本，更新网络
                if all(len(madqn.buffers[i]) >= madqn.mini_batch_size for i in range(madqn.num_agents)):
                    madqn.update()

                # 更新目标网络
                if step_count % madqn.update_freq == 0:
                    madqn.update_target_qnet()

                # 更新状态
                states = next_states
                step_count += 1

                # 检查是否所有agent都完成了
                if all(dones):
                    break

            # epsilon按episode衰减（每个agent独立）
            for agent_id in range(madqn.num_agents):
                madqn.epsilons[agent_id] = max(
                    madqn.epsilon_min,
                    madqn.epsilons[agent_id] * madqn.epsilon_decay
                )

            # 进度条显示信息，分别显示每个agent的return
            postfix_dict = {'Episode': ep}
            for agent_id in range(madqn.num_agents):
                postfix_dict[f'R{agent_id}'] = f'{ep_returns[agent_id]:.2f}'
            avg_epsilon = sum(madqn.epsilons) / len(madqn.epsilons)
            postfix_dict['AvgEpsilon'] = f'{avg_epsilon:.3f}'
            pbar.set_postfix(postfix_dict)

        # 关闭当前iteration的进度条，准备下一个iteration
        pbar.close()
        print(f"Iteration {i+1}/{madqn.iteration} 完成")

    print(f"\n{'='*60}")
    print(f"所有训练完成！总迭代次数: {madqn.iteration}")
    print(f"{'='*60}")


if __name__ == "__main__":
    env = Env()
    madqn = MADQN(
        env,
        lr=0.001, gamma=0.99, iteration=5,
        epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.95,
        num_episodes=50, episode_length=4000,
        batch_size=64, mini_batch_size=32,
        hidden_dim=128, update_freq=10
    )

    train_madqn(madqn)
    madqn.save("models/madqn_model.pth")
