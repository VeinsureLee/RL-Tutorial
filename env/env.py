import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config.env_arguments import env_parser
from PIL import Image
import io
from tqdm import tqdm


# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Env:
    def __init__(self, env_parser=env_parser):
        self.traj = []  # 存储所有agent的轨迹，traj[i]是第i个agent的轨迹列表
        self.render_frames = []  # 存储每一帧的渲染数据

        # 获取环境参数
        args = env_parser.parse_args()
        self.map_size = np.array(args.map_size, dtype=np.float64)  # 地图实际尺寸
        self.grid_size = float(args.grid_size)  # 网格大小
        
        # 将地图按照grid_size离散化，计算网格数量（使用round避免浮点误差）
        self.grid_rows = int(round(self.map_size[0] / self.grid_size))
        self.grid_cols = int(round(self.map_size[1] / self.grid_size))
        self.size = (self.grid_rows, self.grid_cols)
        self.x_dim = self.grid_rows
        self.y_dim = self.grid_cols
        self.state_num = self.x_dim * self.y_dim
        self.action_dim = len(args.action_space)

        # 将连续坐标转换为离散网格坐标
        start_state_raw = self._continuous_to_discrete(args.start_states)
        target_state_raw = self._continuous_to_discrete(args.target_state)  # 注意：参数名是target_state（单数）
        
        # 处理多agent情况：保存所有agent的起始和目标状态
        if isinstance(start_state_raw, list):
            self.start_states = start_state_raw  # 所有agent的起始状态列表
            self.num_agents = len(start_state_raw)
        else:
            # 单个agent的情况（向后兼容）
            self.start_states = [start_state_raw]
            self.num_agents = 1
            
        if isinstance(target_state_raw, list):
            self.target_states = target_state_raw  # 所有agent的目标状态列表
        else:
            # 单个agent的情况（向后兼容）
            self.target_states = [target_state_raw]
            
        # 为了向后兼容，保留单个agent的属性
        self.start_state = self.start_states[0] if len(self.start_states) > 0 else None
        self.target_state = self.target_states[0] if len(self.target_states) > 0 else None
            
        self.forbidden_states = self._process_forbidden_states(args.forbidden_areas)

        # 所有agent的当前状态
        self.agent_states = self.start_states.copy()
        self.num_actions = len(args.action_space)
        self.action_space = args.action_space

        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

    def _continuous_to_discrete(self, state):
        """
        将连续坐标转换为离散网格坐标，处理浮点运算问题
        """
        if isinstance(state, (list, tuple, np.ndarray)):
            # 检查是否是单个坐标对（两个数字）还是状态列表
            if len(state) == 2:
                # 检查第一个元素是否是数字（单个坐标对）
                first_elem = state[0]
                if isinstance(first_elem, (int, float, np.integer, np.floating)):
                    # 单个状态 (x, y)
                    x, y = state
                    # 将连续坐标除以grid_size并四舍五入，确保是整数
                    grid_x = int(round(float(x) / self.grid_size))
                    grid_y = int(round(float(y) / self.grid_size))
                    # 确保在有效范围内
                    grid_x = max(0, min(grid_x, self.grid_rows - 1))
                    grid_y = max(0, min(grid_y, self.grid_cols - 1))
                    return (grid_x, grid_y)
                else:
                    # 状态列表，递归处理每个状态
                    return [self._continuous_to_discrete(s) for s in state]
            else:
                # 多个状态（长度不为2的列表）
                return [self._continuous_to_discrete(s) for s in state]
        return state

    def _process_forbidden_states(self, forbidden_areas):
        """
        将禁止区域转换为离散网格坐标集合
        """
        forbidden_states = set()
        if isinstance(forbidden_areas, (list, tuple)):
            for area in forbidden_areas:
                if isinstance(area, (tuple, list)) and len(area) == 2:
                    pos, size = area
                    # 将连续坐标转换为离散坐标
                    grid_pos = self._continuous_to_discrete(pos)
                    grid_size = int(round(float(size) / self.grid_size))
                    # 添加该区域内的所有网格
                    for i in range(grid_size):
                        for j in range(grid_size):
                            x = grid_pos[0] + i
                            y = grid_pos[1] + j
                            if 0 <= x < self.grid_rows and 0 <= y < self.grid_cols:
                                forbidden_states.add((x, y))
        return forbidden_states

    # Reset the environment to the start state
    def reset(self):
        # 重置所有agent的状态
        self.agent_states = [state for state in self.start_states]
        # 为每个agent初始化轨迹
        self.traj = [[state] for state in self.start_states]
        self.render_frames = []
        # 返回所有agent的状态
        return self.agent_states.copy(), {}

    # Take a step in the environment
    # actions可以是单个动作（单个agent）或动作列表（多个agent）
    def step(self, actions):
        # 如果actions是单个动作（tuple且长度为2，且元素是数字），转换为列表（向后兼容）
        if isinstance(actions, tuple) and len(actions) == 2:
            # 检查是否是坐标对（动作）还是列表
            if isinstance(actions[0], (int, float, np.integer, np.floating)) and isinstance(actions[1], (int, float, np.integer, np.floating)):
                # 这是一个动作tuple，转换为列表
                actions = [actions]
        elif not isinstance(actions, (list, np.ndarray)):
            # 其他非列表类型也转换为列表
            actions = [actions]
        
        assert len(actions) == self.num_agents, f"动作数量 {len(actions)} 与agent数量 {self.num_agents} 不匹配"
        
        next_states = []
        rewards = []
        dones = []
        
        # 为每个agent执行动作
        for agent_id in range(self.num_agents):
            action = actions[agent_id]
            assert action in self.action_space, f"Agent {agent_id} 的无效动作: {action}. Must be in {self.action_space}"
            
            current_state = self.agent_states[agent_id]
            target_state = self.target_states[agent_id]
            
            # 根据action选择移动方向
            next_state, reward = self._get_next_state_and_reward(current_state, action, target_state)
            done = self._is_done(next_state, target_state)

            # 更新agent状态（确保是整数，避免浮点误差）
            next_state = (int(next_state[0]), int(next_state[1]))
            self.agent_states[agent_id] = next_state
            
            # 记录轨迹
            self.traj[agent_id].append(next_state)
            
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        
        # 如果只有一个agent，返回单个值（向后兼容）
        if self.num_agents == 1:
            return next_states[0], rewards[0], dones[0], {}
        else:
            return next_states, rewards, dones, {}

    # Determine the next state and reward based on current state and action
    def _get_next_state_and_reward(self, state, action, target_state):
        """
        根据当前状态和动作计算下一个状态和奖励
        注意处理浮点运算问题，确保状态始终是整数网格坐标
        :param state: 当前状态
        :param action: 动作
        :param target_state: 目标状态（不同agent可能有不同目标）
        """
        x, y = int(state[0]), int(state[1])  # 确保是整数
        
        # 根据action移动方向计算新状态
        dx, dy = action
        new_x = x + dx
        new_y = y + dy

        # 边界检查
        if new_y >= self.grid_cols:  # down
            new_y = self.grid_cols - 1
            reward = self.reward_forbidden
        elif new_x >= self.grid_rows:  # right
            new_x = self.grid_rows - 1
            reward = self.reward_forbidden
        elif new_y < 0:  # up
            new_y = 0
            reward = self.reward_forbidden
        elif new_x < 0:  # left
            new_x = 0
            reward = self.reward_forbidden
        elif (new_x, new_y) == target_state:  # 到达目标
            reward = self.reward_target
        elif (new_x, new_y) in self.forbidden_states:  # 进入禁止区域
            # 保持在原位置
            new_x, new_y = x, y
            reward = self.reward_forbidden
        else:
            reward = self.reward_step

        # 确保返回的是整数坐标
        return (int(new_x), int(new_y)), reward

    # Check if the current state is the target state
    def _is_done(self, state, target_state):
        return state == target_state

    def render(self, mode='human', save_path=None):
        """
        渲染环境，显示agent的移动轨迹
        :param mode: 'human' 显示图像, 'rgb_array' 返回RGB数组, 'save' 保存为文件
        :param save_path: 如果mode='save'，指定保存路径
        :return: 如果mode='rgb_array'，返回RGB数组；否则返回None
        """
        # 创建地图数组（与visualize_agent.py一致：1表示可通行，0表示禁止区域）
        map_array = np.ones((self.grid_rows, self.grid_cols))
        
        # 标记禁止区域
        for x, y in self.forbidden_states:
            if 0 <= x < self.grid_rows and 0 <= y < self.grid_cols:
                map_array[x, y] = 0
        
        # 创建图形（与visualize_agent.py一致的尺寸比例）
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 显示地图（与visualize_agent.py一致：使用'gray' colormap和origin='lower'）
        im = ax.imshow(map_array, cmap='gray', origin='lower', 
                      extent=[0, self.grid_cols, 0, self.grid_rows],
                      interpolation='nearest')
        
        # 添加colorbar（与visualize_agent.py一致）
        plt.colorbar(im, ax=ax, label="Forbidden Area (0 = forbidden)")
        
        # 定义不同agent的颜色
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # 标记所有agent的起始位置
        for i, start_state in enumerate(self.start_states):
            if isinstance(start_state, (list, tuple)) and len(start_state) == 2:
                start_x, start_y = start_state
                color = colors[i % len(colors)]
                ax.scatter([start_y], [start_x], c=color, marker='o', s=100, 
                          edgecolors='black', linewidths=1, zorder=5)
                ax.text(start_y, start_x, f'start{i+1}', fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # 标记所有agent的目标位置
        for i, target_state in enumerate(self.target_states):
            if isinstance(target_state, (list, tuple)) and len(target_state) == 2:
                target_x, target_y = target_state
                color = colors[i % len(colors)]
                ax.scatter([target_y], [target_x], c=color, marker='*', s=180, 
                          edgecolors='black', linewidths=1, zorder=5)
                ax.text(target_y, target_x, f'target{i+1}', fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        # 绘制所有agent的轨迹
        for i, agent_traj in enumerate(self.traj):
            if len(agent_traj) > 0:
                color = colors[i % len(colors)]
                traj_array = np.array(agent_traj)
                ax.plot(traj_array[:, 1], traj_array[:, 0], '-', color=color, linewidth=2, 
                       alpha=0.6, label=f'Agent {i+1}', zorder=3)
                # 标记当前agent位置
                if len(agent_traj) > 0:
                    current_pos = agent_traj[-1]
                    ax.scatter([current_pos[1]], [current_pos[0]], c=color, 
                              marker='s', s=80, edgecolors='black', 
                              linewidths=1, zorder=6)
        
        ax.set_xlabel('Y coordinate', fontsize=12)
        ax.set_ylabel('X coordinate', fontsize=12)
        ax.set_title('Map with Forbidden Areas and Agent States', fontsize=14)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if mode == 'human':
            plt.show()
        elif mode == 'save' and save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        elif mode == 'rgb_array':
            fig.canvas.draw()
            rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return rgb_array
        else:
            plt.close()
        
        return None

    def render_animation(self, interval=200, save_path=None, max_frames=100):
        """
        生成动画，按帧显示agent的移动过程
        :param interval: 每帧之间的时间间隔（毫秒）
        :param save_path: 如果提供，保存动画为gif文件
        :param max_frames: 保存GIF时的最大帧数，超过此数量将抽取关键帧（默认1000）
        :return: FuncAnimation对象
        """
        if len(self.traj) == 0:
            print("警告: 没有轨迹数据，请先运行step()方法")
            return None
        
        # 创建地图数组（与visualize_agent.py一致）
        map_array = np.ones((self.grid_rows, self.grid_cols))
        for x, y in self.forbidden_states:
            if 0 <= x < self.grid_rows and 0 <= y < self.grid_cols:
                map_array[x, y] = 0
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 显示地图（与visualize_agent.py一致：使用'gray' colormap和origin='lower'）
        im = ax.imshow(map_array, cmap='gray', origin='lower',
                      extent=[0, self.grid_cols, 0, self.grid_rows],
                      interpolation='nearest')
        
        # 添加colorbar（与visualize_agent.py一致）
        plt.colorbar(im, ax=ax, label="Forbidden Area (0 = forbidden)")
        
        # 定义不同agent的颜色
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # 标记所有agent的起始和目标位置
        for i, (start_state, target_state) in enumerate(zip(self.start_states, self.target_states)):
            color = colors[i % len(colors)]
            if isinstance(start_state, (list, tuple)) and len(start_state) == 2:
                start_x, start_y = start_state
                ax.scatter([start_y], [start_x], c=color, marker='o', s=100,
                          edgecolors='black', linewidths=1, zorder=5)
                ax.text(start_y, start_x, f'start{i+1}', fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            
            if isinstance(target_state, (list, tuple)) and len(target_state) == 2:
                target_x, target_y = target_state
                ax.scatter([target_y], [target_x], c=color, marker='*', s=180,
                          edgecolors='black', linewidths=1, zorder=5)
                ax.text(target_y, target_x, f'target{i+1}', fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        # 初始化所有agent的轨迹线和位置
        lines = []
        points = []
        for i in range(self.num_agents):
            color = colors[i % len(colors)]
            line, = ax.plot([], [], '-', color=color, linewidth=2, alpha=0.6, zorder=3)
            point, = ax.plot([], [], 's', color=color, markersize=8, zorder=6)
            lines.append(line)
            points.append(point)
        
        ax.set_xlabel('Y coordinate', fontsize=12)
        ax.set_ylabel('X coordinate', fontsize=12)
        ax.set_xlim(-0.5, self.grid_cols - 0.5)
        ax.set_ylim(-0.5, self.grid_rows - 0.5)
        
        # 计算最大轨迹长度
        max_traj_length = max([len(traj) for traj in self.traj]) if self.traj else 0
        
        # 创建标题文本对象，使用ax.text以便在blit模式下更新
        # 位置在axes上方，使用axes坐标转换
        title_text = ax.text(0.5, 1.05, 'Agent Movement Animation', 
                            transform=ax.transAxes, ha='center', va='bottom', 
                            fontsize=14, weight='bold')
        
        def animate(frame):
            # 更新所有agent的轨迹
            for i, agent_traj in enumerate(self.traj):
                if frame < len(agent_traj):
                    traj_so_far = agent_traj[:frame+1]
                    traj_array = np.array(traj_so_far)
                    lines[i].set_data(traj_array[:, 1], traj_array[:, 0])
                    
                    # 更新agent当前位置
                    current_pos = traj_so_far[-1]
                    points[i].set_data([current_pos[1]], [current_pos[0]])
                elif len(agent_traj) > 0:
                    # 如果这个agent已经完成，显示完整轨迹和最终位置
                    traj_array = np.array(agent_traj)
                    lines[i].set_data(traj_array[:, 1], traj_array[:, 0])
                    final_pos = agent_traj[-1]
                    points[i].set_data([final_pos[1]], [final_pos[0]])
            
            # 更新标题文本
            title_text.set_text(f'Agent Movement Animation (Step: {frame+1}/{max_traj_length})')
            
            # 返回所有需要更新的对象（包括标题）
            return lines + points + [title_text]
        
        anim = FuncAnimation(fig, animate, frames=max_traj_length, 
                           interval=interval, blit=True, repeat=True)
        
        if save_path:
            # 如果帧数超过max_frames，使用关键帧抽取
            if max_traj_length > max_frames:
                print(f"总帧数 {max_traj_length} 超过最大帧数 {max_frames}，开始抽取关键帧...")
                # 抽取关键帧：第一帧、最后一帧，以及中间均匀采样
                key_frames = [0]  # 第一帧
                if max_traj_length > 1:
                    # 中间均匀采样
                    step = (max_traj_length - 1) / (max_frames - 1)
                    for i in range(1, max_frames - 1):
                        frame_idx = int(round(i * step))
                        if frame_idx < max_traj_length and frame_idx not in key_frames:
                            key_frames.append(frame_idx)
                    # 最后一帧
                    if max_traj_length - 1 not in key_frames:
                        key_frames.append(max_traj_length - 1)
                
                print(f"抽取了 {len(key_frames)} 个关键帧（从 {max_traj_length} 帧中）")
                
                # 渲染关键帧并保存为GIF
                frames = []
                print("正在渲染关键帧...")
                for frame_idx in tqdm(key_frames, desc="渲染进度", total=len(key_frames)):
                    # 更新动画到指定帧
                    animate(frame_idx)
                    fig.canvas.draw()
                    
                    # 将matplotlib图形转换为PIL Image
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    img = Image.open(buf)
                    frames.append(img.copy())
                    buf.close()
                
                # 保存为GIF
                if len(frames) > 0:
                    print("正在保存GIF文件...")
                    frames[0].save(
                        save_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=interval,  # 每帧持续时间（毫秒）
                        loop=0  # 无限循环
                    )
                    print(f"GIF已保存到: {save_path} (共 {len(frames)} 帧)")
            else:
                # 帧数不多，使用原来的方法
                print(f"正在保存GIF文件（共 {max_traj_length} 帧）...")
                anim.save(save_path, writer='pillow', fps=1000//interval)
                print(f"GIF已保存到: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return anim


if __name__ == "__main__":
    env = Env()
    env.reset()
    print(len(env.forbidden_states))
    print(env.forbidden_states)
    print("after reset", env.agent_states)
    print("after reset", len(env.traj[0]))
    env.step([(1, 0), (0, 1), (0, -1), (-1, 0)])
    print("after step", env.agent_states)
    print("after step", len(env.traj[0]))