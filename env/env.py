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
        
        # 处理多agent情况：保存所有agent的起始和目标状态为 (num_agents, 2) 数组
        if isinstance(start_state_raw, list):
            self.start_states = np.array(start_state_raw, dtype=np.int32)
            self.num_agents = len(start_state_raw)
        else:
            self.start_states = np.array([start_state_raw], dtype=np.int32)
            self.num_agents = 1

        if isinstance(target_state_raw, list):
            self.target_states = np.array(target_state_raw, dtype=np.int32)
        else:
            self.target_states = np.array([target_state_raw], dtype=np.int32)

        # 为了向后兼容，保留单个agent的属性
        self.start_state = tuple(self.start_states[0]) if self.num_agents > 0 else None
        self.target_state = tuple(self.target_states[0]) if self.num_agents > 0 else None

        self.forbidden_states = self._process_forbidden_states(args.forbidden_areas)

        # 所有agent的当前状态 (num_agents, 2)
        self.agent_states = self.start_states.copy()
        self.num_actions = len(args.action_space)
        self.action_space = args.action_space

        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

        # 轨迹预分配 (num_agents, max_steps, 2)，用于完全向量化
        self.max_steps = 100000
        self.step_count = 0

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
        将禁止区域转换为离散网格坐标数组 (n, 2)，用于向量化计算
        """
        out = []
        if isinstance(forbidden_areas, (list, tuple)):
            for area in forbidden_areas:
                if isinstance(area, (tuple, list)) and len(area) == 2:
                    pos, size = area
                    grid_pos = self._continuous_to_discrete(pos)
                    grid_size = int(round(float(size) / self.grid_size))
                    xs = np.arange(grid_size, dtype=np.int32) + grid_pos[0]
                    ys = np.arange(grid_size, dtype=np.int32) + grid_pos[1]
                    xx, yy = np.meshgrid(xs, ys, indexing='ij')
                    in_bounds = (xx >= 0) & (xx < self.grid_rows) & (yy >= 0) & (yy < self.grid_cols)
                    out.append(np.column_stack([xx[in_bounds], yy[in_bounds]]))
        if not out:
            return np.zeros((0, 2), dtype=np.int32)
        return np.vstack(out)

    def reset(self):
        """重置所有 agent 状态与轨迹（完全向量化）"""
        self.agent_states = self.start_states.copy()
        self.step_count = 0
        self.traj = np.zeros((self.num_agents, self.max_steps, 2), dtype=np.int32)
        self.traj[:, 0, :] = self.start_states
        self.render_frames = []
        if self.num_agents == 1:
            return tuple(self.agent_states[0].tolist()), {}
        return self.agent_states.tolist(), {}

    def step(self, actions):
        """
        完全向量化的一步：无 for 循环，所有 agent 批量更新。
        actions: 单个 (dx, dy) 或 list/array of shape (num_agents, 2)
        """
        actions = np.asarray(actions, dtype=np.int32)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, axis=0)
        assert actions.shape[0] == self.num_agents, (
            f"动作数量 {actions.shape[0]} 与 agent 数量 {self.num_agents} 不匹配"
        )

        states = self.agent_states
        raw_next = states + actions

        # 越界检测（clip 前）
        out_of_bounds = (
            (raw_next[:, 0] < 0)
            | (raw_next[:, 0] >= self.grid_rows)
            | (raw_next[:, 1] < 0)
            | (raw_next[:, 1] >= self.grid_cols)
        )
        next_states = np.clip(
            raw_next,
            [0, 0],
            [self.grid_rows - 1, self.grid_cols - 1],
        ).astype(np.int32)

        state_ids = next_states[:, 0] * self.grid_cols + next_states[:, 1]
        target_ids = self.target_states[:, 0] * self.grid_cols + self.target_states[:, 1]
        forbidden_ids = (
            self.forbidden_states[:, 0] * self.grid_cols + self.forbidden_states[:, 1]
            if self.forbidden_states.size > 0
            else np.array([], dtype=np.int64)
        )

        rewards = np.full(self.num_agents, self.reward_step, dtype=np.float64)
        done_mask = np.isin(state_ids, target_ids)
        rewards[done_mask] = self.reward_target

        if forbidden_ids.size > 0:
            forbidden_mask = np.isin(state_ids, forbidden_ids)
            rewards[forbidden_mask] = self.reward_forbidden
            next_states[forbidden_mask] = states[forbidden_mask]

        rewards[out_of_bounds] = self.reward_forbidden
        dones = done_mask.copy()

        self.agent_states = next_states
        self.step_count += 1
        self.traj[:, self.step_count, :] = next_states

        # 统一返回列表格式，便于调用方 enumerate(next_states)
        next_list = [tuple(s.tolist()) for s in next_states]
        return next_list, rewards.tolist(), dones.tolist(), {}

    
    def render(self, mode='human', save_path=None):
        """
        渲染环境，显示agent的移动轨迹
        :param mode: 'human' 显示图像, 'rgb_array' 返回RGB数组, 'save' 保存为文件
        :param save_path: 如果mode='save'，指定保存路径
        :return: 如果mode='rgb_array'，返回RGB数组；否则返回None
        """
        map_array = np.ones((self.grid_rows, self.grid_cols))
        if self.forbidden_states.size > 0:
            valid = (
                (self.forbidden_states[:, 0] >= 0)
                & (self.forbidden_states[:, 0] < self.grid_rows)
                & (self.forbidden_states[:, 1] >= 0)
                & (self.forbidden_states[:, 1] < self.grid_cols)
            )
            xs, ys = self.forbidden_states[valid, 0], self.forbidden_states[valid, 1]
            map_array[xs, ys] = 0

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            map_array, cmap='gray', origin='lower',
            extent=[0, self.grid_cols, 0, self.grid_rows],
            interpolation='nearest',
        )
        plt.colorbar(im, ax=ax, label="Forbidden Area (0 = forbidden)")

        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        n_show = self.step_count + 1

        for i in range(self.num_agents):
            color = colors[i % len(colors)]
            ax.scatter(
                [self.start_states[i, 1]], [self.start_states[i, 0]],
                c=color, marker='o', s=100, edgecolors='black', linewidths=1, zorder=5,
            )
            ax.text(
                self.start_states[i, 1], self.start_states[i, 0], f'start{i+1}',
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
            )
            ax.scatter(
                [self.target_states[i, 1]], [self.target_states[i, 0]],
                c=color, marker='*', s=180, edgecolors='black', linewidths=1, zorder=5,
            )
            ax.text(
                self.target_states[i, 1], self.target_states[i, 0], f'target{i+1}',
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
            )

            traj_slice = self.traj[i, :n_show, :]
            if traj_slice.shape[0] > 0:
                ax.plot(
                    traj_slice[:, 1], traj_slice[:, 0], '-',
                    color=color, linewidth=2, alpha=0.6, label=f'Agent {i+1}', zorder=3,
                )
                ax.scatter(
                    [traj_slice[-1, 1]], [traj_slice[-1, 0]],
                    c=color, marker='s', s=80, edgecolors='black', linewidths=1, zorder=6,
                )
        
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
        生成动画，按帧显示 agent 的移动过程（使用向量化 traj 数组）
        """
        max_traj_length = self.step_count + 1
        if max_traj_length <= 0:
            print("警告: 没有轨迹数据，请先运行 step() 方法")
            return None

        map_array = np.ones((self.grid_rows, self.grid_cols))
        if self.forbidden_states.size > 0:
            valid = (
                (self.forbidden_states[:, 0] >= 0)
                & (self.forbidden_states[:, 0] < self.grid_rows)
                & (self.forbidden_states[:, 1] >= 0)
                & (self.forbidden_states[:, 1] < self.grid_cols)
            )
            xs, ys = self.forbidden_states[valid, 0], self.forbidden_states[valid, 1]
            map_array[xs, ys] = 0

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            map_array, cmap='gray', origin='lower',
            extent=[0, self.grid_cols, 0, self.grid_rows],
            interpolation='nearest',
        )
        plt.colorbar(im, ax=ax, label="Forbidden Area (0 = forbidden)")

        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i in range(self.num_agents):
            color = colors[i % len(colors)]
            ax.scatter(
                [self.start_states[i, 1]], [self.start_states[i, 0]],
                c=color, marker='o', s=100, edgecolors='black', linewidths=1, zorder=5,
            )
            ax.text(
                self.start_states[i, 1], self.start_states[i, 0], f'start{i+1}',
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
            )
            ax.scatter(
                [self.target_states[i, 1]], [self.target_states[i, 0]],
                c=color, marker='*', s=180, edgecolors='black', linewidths=1, zorder=5,
            )
            ax.text(
                self.target_states[i, 1], self.target_states[i, 0], f'target{i+1}',
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
            )

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
        title_text = ax.text(
            0.5, 1.05, 'Agent Movement Animation',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=14, weight='bold',
        )

        def animate(frame):
            for i in range(self.num_agents):
                n_show = min(frame + 1, max_traj_length)
                traj_slice = self.traj[i, :n_show, :]
                if traj_slice.shape[0] > 0:
                    lines[i].set_data(traj_slice[:, 1], traj_slice[:, 0])
                    points[i].set_data([traj_slice[-1, 1]], [traj_slice[-1, 0]])
            title_text.set_text(
                f'Agent Movement Animation (Step: {frame+1}/{max_traj_length})',
            )
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
