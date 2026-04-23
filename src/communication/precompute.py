"""
预计算无线电地图：对 120x60 网格一次性计算所有静态无线量并缓存为 .npz 文件。
每步只需查表 + Rayleigh 衰落，省掉距离/角度/PL/LOS 计算。
"""
import os
import hashlib
import numpy as np
from utils.path_tool import get_abs_path


class PrecomputedRadioMap:
    """
    预计算并缓存所有与位置相关的静态无线量。

    缓存内容 (均为 numpy 数组):
        distances:       (rows, cols) float64  — 每个网格到 AP 的 3D 距离 (m)
        aoa:             (rows, cols) float64  — 每个网格到 AP 的到达角 (rad)
        los_grid:        (rows, cols) bool     — LOS=True, NLOS=False
        path_loss:       (rows, cols) float64  — 路径损耗 PL (dB)
        beta:            (rows, cols) float64  — 大尺度衰落系数 10^(-PL/20)
        steering_vectors:(rows, cols, N_t) complex128 — ULA 阵列响应向量

    用法:
        rm = PrecomputedRadioMap(map_size, grid_size, antenna_pos, ...)
        # 给定 K 个 agent 的网格坐标 [(r1,c1), (r2,c2), ...]
        H = rm.get_channel_matrix(positions)  # (K, N_t) complex
    """

    def __init__(
        self,
        map_size: tuple,          # (rows, cols) 网格数，如 (120, 60)
        grid_size: float,         # 网格物理尺寸 (m)，如 0.4
        antenna_position: tuple,  # AP 网格坐标 (x, y)
        h_AP: float,              # AP 高度 (m)
        h_robot: float,           # 机器人天线高度 (m)
        h_block: float,           # 障碍物高度 (m)
        n_antenna: int,           # 天线数 N_t
        carrier_freq_ghz: float,  # 载波频率 (GHz)
        forbidden_areas: list,    # 禁区列表 [(r,c,w,h), ...]
        sigma_rayleigh: float = 1.2,
        cache_dir: str = None,
    ):
        self.map_size = tuple(map_size)
        self.rows, self.cols = self.map_size
        self.grid_size = grid_size
        self.ap_grid = tuple(antenna_position)
        self.ap_pos_m = (self.ap_grid[0] * grid_size, self.ap_grid[1] * grid_size, h_AP)
        self.h_robot = h_robot
        self.h_block = h_block
        self.n_antenna = n_antenna
        self.carrier_freq_ghz = carrier_freq_ghz
        self.forbidden_areas = forbidden_areas
        self.sigma_rayleigh = sigma_rayleigh

        # 波长和天线间距
        self.wavelength = 3e8 / (carrier_freq_ghz * 1e9)
        self.antenna_spacing = self.wavelength / 2.0

        # 缓存路径
        if cache_dir is None:
            cache_dir = get_abs_path("config/dynamic")
        self._cache_path = os.path.join(cache_dir, "radio_map_cache.npz")

        # 计算参数指纹用于缓存失效判断
        self._param_hash = self._compute_param_hash()

        # 加载或计算
        if self._try_load_cache():
            pass  # 缓存命中
        else:
            self._compute_all()
            self._save_cache()

    def _compute_param_hash(self) -> str:
        """根据所有影响预计算结果的参数生成哈希，参数变化时自动重算。"""
        parts = [
            str(self.map_size), str(self.grid_size),
            str(self.ap_grid), str(self.ap_pos_m),
            str(self.h_robot), str(self.h_block),
            str(self.n_antenna), str(self.carrier_freq_ghz),
            str(sorted([tuple(a) for a in self.forbidden_areas])),
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _try_load_cache(self) -> bool:
        """尝试从 .npz 加载缓存，校验参数哈希。"""
        if not os.path.isfile(self._cache_path):
            return False
        try:
            data = np.load(self._cache_path, allow_pickle=True)
            if str(data.get("param_hash", "")) != self._param_hash:
                return False
            self.distances = data["distances"]
            self.aoa = data["aoa"]
            self.los_grid = data["los_grid"]
            self.path_loss = data["path_loss"]
            self.beta = data["beta"]
            self.steering_vectors = data["steering_vectors"]
            return True
        except Exception:
            return False

    def _save_cache(self):
        """保存预计算结果到 .npz。"""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        np.savez_compressed(
            self._cache_path,
            param_hash=np.array(self._param_hash),
            distances=self.distances,
            aoa=self.aoa,
            los_grid=self.los_grid,
            path_loss=self.path_loss,
            beta=self.beta,
            steering_vectors=self.steering_vectors,
        )

    def _compute_all(self):
        """一次性计算所有 120x60 网格的静态无线量。"""
        rows, cols = self.rows, self.cols
        gs = self.grid_size
        ap_x, ap_y, ap_z = self.ap_pos_m

        # 1. 距离和角度
        # 网格中心的物理坐标
        grid_x = (np.arange(rows) + 0.5) * gs  # (rows,)
        grid_y = (np.arange(cols) + 0.5) * gs   # (cols,)
        gx, gy = np.meshgrid(grid_x, grid_y, indexing='ij')  # (rows, cols)

        dx = gx - ap_x
        dy = gy - ap_y
        dz = self.h_robot - ap_z  # 常数

        dist_2d = np.sqrt(dx**2 + dy**2)
        self.distances = np.sqrt(dx**2 + dy**2 + dz**2)  # 3D 距离
        self.distances = np.maximum(self.distances, 0.1)  # 避免 log(0)

        # 到达角 (azimuth)
        self.aoa = np.arctan2(dy, dx)  # (rows, cols)

        # 2. LOS/NLOS 判定 (AABB)
        self.los_grid = self._compute_los_grid(gx, gy)

        # 3. 路径损耗
        fc = self.carrier_freq_ghz
        d = self.distances
        pl_los = 31.84 + 21.50 * np.log10(d) + 19.00 * np.log10(fc)
        pl_sl = 33.0 + 25.50 * np.log10(d) + 20.00 * np.log10(fc)
        pl_nlos = np.maximum(pl_sl, pl_los)

        self.path_loss = np.where(self.los_grid, pl_los, pl_nlos)

        # 4. 大尺度衰落系数
        self.beta = np.power(10.0, -self.path_loss / 20.0)

        # 5. ULA 阵列响应向量
        n = self.n_antenna
        sin_aoa = np.sin(self.aoa)  # (rows, cols)
        # 向量化：对每个网格位置生成长度为 n 的阵列响应
        antenna_indices = np.arange(n)  # (n,)
        # phase: (rows, cols, n)
        phase = -1j * np.pi * sin_aoa[:, :, np.newaxis] * antenna_indices[np.newaxis, np.newaxis, :]
        self.steering_vectors = np.exp(phase) / np.sqrt(n)  # (rows, cols, n)

    def _compute_los_grid(self, gx, gy):
        """AABB 射线-方框相交检测，判断每个网格到 AP 是否被障碍物遮挡。"""
        rows, cols = self.rows, self.cols
        gs = self.grid_size
        ap_x, ap_y = self.ap_pos_m[0], self.ap_pos_m[1]

        los = np.ones((rows, cols), dtype=bool)

        for area in self.forbidden_areas:
            # area 格式: (row, col, width, height) 或 ((row, col), size)
            if len(area) == 4:
                r, c, w, h = area
            elif len(area) == 2 and isinstance(area[0], (list, tuple, np.ndarray)):
                pos, size = area[0], area[1]
                r, c = pos[0], pos[1]
                if isinstance(size, (int, float, np.integer)):
                    w, h = int(size), int(size)
                else:
                    w, h = size[0] - pos[0], size[1] - pos[1]
            else:
                continue

            # 障碍物物理边界
            box_x_min = r * gs
            box_x_max = (r + w) * gs
            box_y_min = c * gs
            box_y_max = (c + h) * gs

            # 射线方向: 从 (gx, gy) 到 (ap_x, ap_y)
            dir_x = ap_x - gx
            dir_y = ap_y - gy

            # AABB 射线相交 (slab method)
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_dir_x = np.where(np.abs(dir_x) > 1e-10, 1.0 / dir_x, np.inf)
                inv_dir_y = np.where(np.abs(dir_y) > 1e-10, 1.0 / dir_y, np.inf)

                tx1 = (box_x_min - gx) * inv_dir_x
                tx2 = (box_x_max - gx) * inv_dir_x
                ty1 = (box_y_min - gy) * inv_dir_y
                ty2 = (box_y_max - gy) * inv_dir_y

            tmin_x = np.minimum(tx1, tx2)
            tmax_x = np.maximum(tx1, tx2)
            tmin_y = np.minimum(ty1, ty2)
            tmax_y = np.maximum(ty1, ty2)

            tmin = np.maximum(tmin_x, tmin_y)
            tmax = np.minimum(tmax_x, tmax_y)

            # 相交条件: tmin <= tmax 且 tmax > 0 且 tmin < 1
            intersects = (tmin <= tmax) & (tmax > 1e-6) & (tmin < 1.0 - 1e-6)

            los &= ~intersects

        return los

    def get_channel_vectors(self, positions, rng=None):
        """
        给定 K 个 agent 的网格坐标，返回含 Rayleigh 衰落的信道向量矩阵。

        Args:
            positions: list of (row, col) 或 (K, 2) array，网格坐标（整数）
            rng: numpy RandomState，不传则用全局随机

        Returns:
            H: (K, N_t) complex128 信道向量矩阵
        """
        positions = np.array(positions, dtype=int)
        K = len(positions)
        rs = positions[:, 0].clip(0, self.rows - 1)
        cs = positions[:, 1].clip(0, self.cols - 1)

        # 查表: beta 和 steering vector
        betas = self.beta[rs, cs]             # (K,)
        svs = self.steering_vectors[rs, cs]   # (K, N_t)

        # Rayleigh 衰落: CN(0, sigma^2)
        if rng is None:
            rng = np.random.default_rng()
        rayleigh = (
            rng.normal(0, self.sigma_rayleigh, (K, self.n_antenna))
            + 1j * rng.normal(0, self.sigma_rayleigh, (K, self.n_antenna))
        ) / np.sqrt(2)

        # 信道向量 = beta * steering_vector * rayleigh_fading
        H = betas[:, np.newaxis] * svs * rayleigh

        return H

    def get_path_loss_at(self, positions):
        """查表返回路径损耗。positions: (K, 2) int array。"""
        positions = np.array(positions, dtype=int)
        rs = positions[:, 0].clip(0, self.rows - 1)
        cs = positions[:, 1].clip(0, self.cols - 1)
        return self.path_loss[rs, cs]

    def get_los_at(self, positions):
        """查表返回 LOS/NLOS。positions: (K, 2) int array。"""
        positions = np.array(positions, dtype=int)
        rs = positions[:, 0].clip(0, self.rows - 1)
        cs = positions[:, 1].clip(0, self.cols - 1)
        return self.los_grid[rs, cs]
