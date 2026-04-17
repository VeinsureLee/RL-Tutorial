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

    def render_frame(self, ax=None):
        """渲染当前帧到 matplotlib axes，用于 GIF 生成。"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.clear()
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # 禁区
        for (r, c) in self.forbidden_set:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='gray', alpha=0.5))

        # AP
        ap = self.radio_map.ap_grid
        ax.plot(ap[1], ap[0], 'r^', markersize=10, label='AP')

        # Agent 轨迹和当前位置
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
        for i in range(self.num_agents):
            traj = self.trajectories[i]
            if len(traj) > 1:
                traj_arr = np.array(traj)
                ax.plot(traj_arr[:, 1], traj_arr[:, 0], '-', color=colors[i], alpha=0.5, linewidth=1)

            # 当前位置
            r, c = self.positions[i]
            marker = '*' if self.done_flags[i] else 'o'
            ax.plot(c, r, marker, color=colors[i], markersize=8, label=f'Agent {i}')

            # 目标
            tr, tc = self.target_states[i]
            ax.plot(tc, tr, 'x', color=colors[i], markersize=8)

        ax.legend(loc='upper right', fontsize=6)
        return ax

    def save_gif(self, path, frames_data, fps=5):
        """
        从帧数据列表生成 GIF。

        Args:
            path: 保存路径
            frames_data: list of (positions, done_flags, trajectories) tuples
            fps: 帧率
        """
        images = []
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for positions, done_flags, trajectories in frames_data:
            ax.clear()
            ax.set_xlim(-0.5, self.cols - 0.5)
            ax.set_ylim(-0.5, self.rows - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()

            for (r, c) in self.forbidden_set:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='gray', alpha=0.5))

            ap = self.radio_map.ap_grid
            ax.plot(ap[1], ap[0], 'r^', markersize=10)

            colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
            for i in range(self.num_agents):
                traj = trajectories[i]
                if len(traj) > 1:
                    traj_arr = np.array(traj)
                    ax.plot(traj_arr[:, 1], traj_arr[:, 0], '-', color=colors[i], alpha=0.5)
                r, c = positions[i]
                marker = '*' if done_flags[i] else 'o'
                ax.plot(c, r, marker, color=colors[i], markersize=8)
                tr, tc = self.target_states[i]
                ax.plot(tc, tr, 'x', color=colors[i], markersize=8)

            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            images.append(Image.open(buf).copy())
            buf.close()

        plt.close(fig)

        if images:
            duration = int(1000 / fps)
            images[0].save(path, save_all=True, append_images=images[1:],
                          duration=duration, loop=0)
