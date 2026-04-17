"""
多机器人网格导航环境（对齐论文）。
120x60 网格 (48m x 24m, grid=0.4m)，动作空间 = 4方向 x p功率等级。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from communication.precompute import PrecomputedRadioMap
from communication.ber_reward import compute_ber_rewards


class MultiRobotEnv:
    """
    多机器人通感协同导航环境。

    观测: agent 的网格坐标 (row, col) -> 单个整数 state_index
    动作: 0 ~ (n_dirs * n_powers - 1) 的整数，解码为 (方向, 功率等级)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict，包含以下 key：
                map_size, grid_size, antenna_position,
                start_states, target_states, forbidden_areas,
                action_directions, reward_goal, reward_closer,
                reward_farther, reward_same, reward_forbidden, omega,
                h_AP, h_robot, h_block,
                n_antenna, carrier_freq_ghz, sigma_rayleigh,
                P_sum, P_min_diff, num_power_levels,
                channel_block_length, packet_size, noise_power_mw
        """
        # 地图
        self.map_size = tuple(int(x) for x in config["map_size"])
        self.rows, self.cols = self.map_size
        self.grid_size = config["grid_size"]
        self.n_states = self.rows * self.cols

        # 兼容旧接口
        self.x_dim = self.rows
        self.y_dim = self.cols
        self.state_num = self.n_states

        # Agent 起终点
        self.start_states = [tuple(int(x) for x in s) for s in config["start_states"]]
        self.target_states = [tuple(int(x) for x in s) for s in config["target_states"]]
        self.num_agents = len(self.start_states)

        # 禁区 -> 占用网格集合
        self.forbidden_areas_raw = config["forbidden_areas"]
        self.forbidden_set = self._build_forbidden_set(self.forbidden_areas_raw)

        # 动作空间
        self.directions = [tuple(d) for d in config["action_directions"]]
        self.n_dirs = len(self.directions)
        self.n_powers = config["num_power_levels"]
        self.n_actions = self.n_dirs * self.n_powers
        self.num_actions = self.n_actions  # 兼容旧接口

        # 奖励参数（论文对齐）
        self.reward_goal = config["reward_goal"]
        self.reward_closer = config["reward_closer"]      # -0.8
        self.reward_farther = config["reward_farther"]     # 0.1
        self.reward_same = config["reward_same"]           # 0.0
        self.reward_forbidden = config["reward_forbidden"]
        self.omega = config["omega"]

        # 通信参数
        self.P_sum = config["P_sum"]
        self.N = config["channel_block_length"]
        self.D = config["packet_size"]
        self.noise_power = config["noise_power_mw"]

        # 预计算无线电地图
        self.radio_map = PrecomputedRadioMap(
            map_size=self.map_size,
            grid_size=self.grid_size,
            antenna_position=tuple(config["antenna_position"]),
            h_AP=config["h_AP"],
            h_robot=config["h_robot"],
            h_block=config["h_block"],
            n_antenna=config["n_antenna"],
            carrier_freq_ghz=config["carrier_freq_ghz"],
            forbidden_areas=self.forbidden_areas_raw,
            sigma_rayleigh=config.get("sigma_rayleigh", 1.2),
        )

        # 随机数生成器
        self.rng = np.random.default_rng(config.get("random_seed", 42))

        # 状态
        self.positions = None    # (num_agents, 2) int array
        self.done_flags = None   # (num_agents,) bool
        self.trajectories = None # list of lists

        print("---------- 环境加载 ----------")
        print(f"  agent num:     {self.num_agents}")
        print(f"  grid (rows x cols): {self.rows} x {self.cols}")
        print(f"  n_actions:     {self.n_actions} ({self.n_dirs} dirs x {self.n_powers} powers)")
        print(f"  omega:         {self.omega}")
        print("------------------------------")

    def _build_forbidden_set(self, areas):
        """将禁区列表转为网格坐标集合。"""
        forbidden = set()
        for area in areas:
            if isinstance(area, (list, tuple)):
                if len(area) == 4:
                    r, c, w, h = area
                    for dr in range(int(w)):
                        for dc in range(int(h)):
                            forbidden.add((int(r + dr), int(c + dc)))
                elif len(area) == 2:
                    first, second = area
                    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
                        # [(r1,c1), (r2,c2)] 范围格式
                        (r1, c1), (r2, c2) = first, second
                        for r in range(int(r1), int(r2)):
                            for c in range(int(c1), int(c2)):
                                forbidden.add((r, c))
                    elif isinstance(first, (list, tuple)) and len(first) == 2:
                        # (pos, size) 格式: pos=(row,col), size=scalar
                        for dr in range(int(second)):
                            for dc in range(int(second)):
                                forbidden.add((int(first[0] + dr), int(first[1] + dc)))
        return forbidden

    def pos_to_index(self, row, col):
        """网格坐标 -> 状态索引。"""
        return int(row) * self.cols + int(col)

    def index_to_pos(self, idx):
        """状态索引 -> 网格坐标。"""
        return (idx // self.cols, idx % self.cols)

    def decode_action(self, action):
        """
        复合动作 -> (方向索引, 功率等级索引)。
        action = dir_idx * n_powers + power_idx
        """
        dir_idx = action // self.n_powers
        power_idx = action % self.n_powers
        return dir_idx, power_idx

    def reset(self):
        """
        重置环境。

        Returns:
            states: list of int，每个 agent 的初始状态索引
        """
        self.positions = np.array(self.start_states, dtype=int)
        self.done_flags = np.zeros(self.num_agents, dtype=bool)
        self.trajectories = [[(int(r), int(c))] for r, c in self.positions]

        return [self.pos_to_index(r, c) for r, c in self.positions]

    def step(self, actions):
        """
        执行一步。

        Args:
            actions: list of int，每个 agent 的复合动作

        Returns:
            next_states: list of int
            rewards: list of float
            dones: list of bool
            info: dict with 'ber', 'sinr', 'power_indices'
        """
        actions = list(actions)
        power_indices = np.zeros(self.num_agents, dtype=int)
        next_positions = self.positions.copy()

        for i in range(self.num_agents):
            if self.done_flags[i]:
                continue

            dir_idx, power_idx = self.decode_action(actions[i])
            power_indices[i] = power_idx

            # 移动
            dr, dc = self.directions[dir_idx]
            new_r = int(self.positions[i, 0] + dr)
            new_c = int(self.positions[i, 1] + dc)

            # 边界检查
            if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                next_positions[i] = [new_r, new_c]

        # 计算奖励
        rewards = np.zeros(self.num_agents)
        active_indices = np.where(~self.done_flags)[0]

        for i in active_indices:
            r, c = next_positions[i]

            # 禁区惩罚
            if (r, c) in self.forbidden_set:
                rewards[i] = self.reward_forbidden
                next_positions[i] = self.positions[i].copy()  # 撤回移动
                continue

            # 到达目标
            if (r, c) == self.target_states[i]:
                rewards[i] = self.reward_goal
                self.done_flags[i] = True
                continue

            # 距离奖励（论文式 3-1）
            old_dist = abs(self.positions[i, 0] - self.target_states[i][0]) + \
                       abs(self.positions[i, 1] - self.target_states[i][1])
            new_dist = abs(r - self.target_states[i][0]) + abs(c - self.target_states[i][1])

            if new_dist < old_dist:
                rewards[i] = self.reward_closer    # -0.8 (论文)
            elif new_dist > old_dist:
                rewards[i] = self.reward_farther   # 0.1 (论文)
            else:
                rewards[i] = self.reward_same      # 0.0

        # 更新位置
        self.positions = next_positions

        # 记录轨迹
        for i in range(self.num_agents):
            self.trajectories[i].append((int(self.positions[i, 0]), int(self.positions[i, 1])))

        # 通信质量计算（只对活跃 agent）
        ber_info = {"ber": np.zeros(self.num_agents), "sinr": np.zeros(self.num_agents),
                    "reward": np.zeros(self.num_agents)}

        if len(active_indices) > 0:
            active_positions = self.positions[active_indices]
            active_powers = power_indices[active_indices]

            ber_result = compute_ber_rewards(
                radio_map=self.radio_map,
                positions=active_positions,
                power_actions=active_powers,
                P_sum=self.P_sum,
                num_power_levels=self.n_powers,
                N=self.N,
                D=self.D,
                noise_power=self.noise_power,
                rng=self.rng,
            )

            for j, idx in enumerate(active_indices):
                ber_info["ber"][idx] = ber_result["ber"][j]
                ber_info["sinr"][idx] = ber_result["sinr"][j]
                ber_info["reward"][idx] = ber_result["reward"][j]

            # 通信奖励加入总奖励
            for idx in active_indices:
                if not self.done_flags[idx] and (int(self.positions[idx, 0]), int(self.positions[idx, 1])) not in self.forbidden_set:
                    rewards[idx] += self.omega * ber_info["reward"][idx]

        # 构造返回值
        next_states = [self.pos_to_index(r, c) for r, c in self.positions]
        dones = self.done_flags.tolist()

        info = {
            "ber": ber_info["ber"],
            "sinr": ber_info["sinr"],
            "power_indices": power_indices,
        }

        return next_states, rewards.tolist(), dones, info

    @property
    def all_done(self):
        """所有 agent 是否都到达目标。"""
        return bool(np.all(self.done_flags))

    # ------------------------------------------------------------------
    # 可视化常量
    # ------------------------------------------------------------------
    AGENT_COLORS = ["blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    # ------------------------------------------------------------------
    # 内部绘图辅助
    # ------------------------------------------------------------------
    def _draw_obstacles(self, ax):
        """在 ax 上画黑色障碍物块。"""
        for (r, c) in self.forbidden_set:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))

    def _draw_agents(self, ax, positions, done_flags, trajectories):
        """在 ax 上画起点(圆)、终点(星)、轨迹线、当前位置(方块)。"""
        colors = self.AGENT_COLORS
        for i in range(self.num_agents):
            c = colors[i % len(colors)]
            # 起点
            sr, sc = self.start_states[i]
            ax.scatter([sc], [sr], c=c, marker='o', s=100, edgecolors='black',
                       linewidths=1, zorder=5)
            # 终点
            tr, tc = self.target_states[i]
            ax.scatter([tc], [tr], c=c, marker='*', s=200, edgecolors='black',
                       linewidths=1, zorder=5)
            # 轨迹
            traj = trajectories[i] if trajectories is not None else self.trajectories[i]
            if len(traj) > 1:
                arr = np.array(traj)
                ax.plot(arr[:, 1], arr[:, 0], '-', color=c, linewidth=2, alpha=0.6, zorder=3)
            # 当前位置
            pos = positions[i] if positions is not None else self.positions[i]
            pr, pc = pos
            done = done_flags[i] if done_flags is not None else self.done_flags[i]
            marker = '*' if done else 's'
            ax.scatter([pc], [pr], c=c, marker=marker, s=80, edgecolors='black',
                       linewidths=1, zorder=6)

    def _setup_ax(self, ax):
        """设置 ax 基本属性。"""
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

    # ------------------------------------------------------------------
    # 导航地图（白色背景 + 黑色障碍物）
    # ------------------------------------------------------------------
    def render_nav_frame(self, ax, positions=None, done_flags=None, trajectories=None):
        """白色背景导航地图：黑色障碍物 + 起点/终点/轨迹。"""
        ax.clear()
        self._setup_ax(ax)
        ax.set_facecolor('white')
        self._draw_obstacles(ax)
        self._draw_agents(ax, positions, done_flags, trajectories)
        ax.set_title("Navigation Map")

    # ------------------------------------------------------------------
    # 通信质量地图（路径损耗热力图背景 + 黑色障碍物）
    # ------------------------------------------------------------------
    def _get_signal_display(self):
        """计算信号质量显示矩阵和统一的 vmin/vmax，供 PNG 和 GIF 共用。"""
        pl = self.radio_map.path_loss.copy()
        display = -pl
        valid = display[np.isfinite(display)]
        vmin_val = float(np.min(valid)) if len(valid) > 0 else -100.0
        for (r, c) in self.forbidden_set:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                display[r, c] = vmin_val - 20
        sig_vmin = float(np.percentile(valid, 1)) if len(valid) > 0 else -100.0
        sig_vmax = float(np.percentile(valid, 99)) if len(valid) > 0 else -40.0
        return display, sig_vmin, sig_vmax

    def _draw_signal_bg(self, ax, display, sig_vmin, sig_vmax):
        """在 ax 上画信号质量热力图背景。"""
        ax.imshow(display, cmap='turbo', origin='upper',
                  extent=[-0.5, self.cols - 0.5, self.rows - 0.5, -0.5],
                  aspect='equal', interpolation='nearest',
                  vmin=sig_vmin, vmax=sig_vmax)

    def render_signal_frame(self, ax, positions=None, done_flags=None, trajectories=None,
                            display=None, sig_vmin=None, sig_vmax=None, colorbar=True):
        """通信质量背景地图：路径损耗热力图 + 黑色障碍物 + 起点/终点/轨迹。"""
        ax.clear()
        self._setup_ax(ax)
        if display is None:
            display, sig_vmin, sig_vmax = self._get_signal_display()
        im = ax.imshow(display, cmap='turbo', origin='upper',
                       extent=[-0.5, self.cols - 0.5, self.rows - 0.5, -0.5],
                       aspect='equal', interpolation='nearest',
                       vmin=sig_vmin, vmax=sig_vmax)
        if colorbar:
            ax.figure.colorbar(im, ax=ax, label='Signal Strength (dB)', shrink=0.8)
        self._draw_obstacles(ax)
        self._draw_agents(ax, positions, done_flags, trajectories)
        ax.set_title("Signal Quality Map")

    # ------------------------------------------------------------------
    # 兼容旧接口
    # ------------------------------------------------------------------
    def render_frame(self, ax=None):
        """渲染当前帧（导航地图）。"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        self.render_nav_frame(ax)
        return ax

    # ------------------------------------------------------------------
    # 双视图：导航 + 通信质量
    # ------------------------------------------------------------------
    def render_dual(self, save_path=None):
        """渲染双视图并保存 PNG。"""
        fig, (ax_nav, ax_sig) = plt.subplots(1, 2, figsize=(20, 8))
        self.render_nav_frame(ax_nav)
        self.render_signal_frame(ax_sig)
        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.AGENT_COLORS[i % len(self.AGENT_COLORS)],
                  edgecolor='black', label=f'Agent {i}')
            for i in range(self.num_agents)
        ]
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=min(self.num_agents, 8), frameon=True)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # GIF 保存（双视图）
    # ------------------------------------------------------------------
    def _sample_frame_indices(self, total, max_frames=100):
        """均匀抽帧，返回帧索引列表。"""
        if total > max_frames:
            indices = [0] + [int(round(i * (total - 1) / (max_frames - 1)))
                             for i in range(1, max_frames)]
            return sorted(set(indices))
        return list(range(total))

    def _render_gif_frames(self, fig, ax, frames_data, indices, render_fn, **render_kwargs):
        """通用 GIF 帧渲染，返回 PIL Image 列表。"""
        total = len(frames_data)
        images = []
        for idx in indices:
            positions, done_flags, trajectories = frames_data[idx]
            render_fn(ax, positions=positions, done_flags=done_flags,
                      trajectories=trajectories, **render_kwargs)
            ax.set_title(ax.get_title().split("(")[0] + f"(Step {idx}/{total - 1})")
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            images.append(Image.open(buf).copy())
            buf.close()
        return images

    def _save_images_as_gif(self, images, path, fps):
        """将 PIL Image 列表保存为 GIF。"""
        if images:
            duration = int(1000 / fps)
            images[0].save(path, save_all=True, append_images=images[1:],
                           duration=duration, loop=0)

    def save_nav_gif(self, path, frames_data, fps=5, max_frames=100):
        """保存导航地图 GIF（白色背景）。"""
        indices = self._sample_frame_indices(len(frames_data), max_frames)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        images = self._render_gif_frames(fig, ax, frames_data, indices, self.render_nav_frame)
        plt.close(fig)
        self._save_images_as_gif(images, path, fps)

    def save_signal_gif(self, path, frames_data, fps=5, max_frames=100):
        """保存通信质量地图 GIF（热力图背景）。"""
        indices = self._sample_frame_indices(len(frames_data), max_frames)
        display, sig_vmin, sig_vmax = self._get_signal_display()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        images = self._render_gif_frames(
            fig, ax, frames_data, indices, self.render_signal_frame,
            display=display, sig_vmin=sig_vmin, sig_vmax=sig_vmax, colorbar=False)
        plt.close(fig)
        self._save_images_as_gif(images, path, fps)
