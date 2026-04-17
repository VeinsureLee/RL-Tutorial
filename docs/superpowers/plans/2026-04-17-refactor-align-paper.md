# Refactor: Align Environment & Communication with Reference Paper

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the project to match the reference thesis paper exactly: 120x60 grid, binary power control in action space, precomputed radio map cache, and aligned hyperparameters.

**Architecture:** Static radio quantities (distance, AoA, LOS/NLOS, path loss, steering vectors) are precomputed once for all 7200 grid cells and cached as `.npz`. Each step only adds Rayleigh fading, clusters, SVD precodes, and computes BER. Action space expands to direction x power_level. All RL hyperparameters align with the paper.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, PIL

---

## File Structure

| Operation | File | Responsibility |
|-----------|------|----------------|
| **Create** | `communication/precompute.py` | PrecomputedRadioMap class: one-time compute & cache of all static radio quantities |
| **Rewrite** | `env/env.py` | 120x60 grid, expanded action space (dir x power), paper-aligned reward function |
| **Rewrite** | `communication/ber_reward.py` | Uses PrecomputedRadioMap, batch channel vector assembly, returns BER vector |
| **Rewrite** | `communication/SIC.py` | Binary power control scheme, SINR/BER with power as action |
| **Modify** | `communication/channel.py` | Add batch methods, keep path_loss functions, remove per-call overhead |
| **Modify** | `communication/diagonalization_precoding.py` | No logic change, just accept pre-shaped matrices |
| **Modify** | `rl_algorithms/net/qnet.py` | Output dim = n_directions x n_power_levels |
| **Modify** | `rl_algorithms/structure/madqn.py` | Decode compound action (direction, power), pass power to env |
| **Modify** | `rl_algorithms/structure/dqn.py` | Same compound action decoding |
| **Modify** | `rl_algorithms/train/train_madqn.py` | Pass power info, new param defaults |
| **Modify** | `rl_algorithms/train/train_dqn.py` | Same |
| **Modify** | `rl_algorithms/train/run.py` | New default params |
| **Modify** | `rl_algorithms/test/run.py` | Adapt to new env API |
| **Modify** | `config/base/env.yml` | 120x60 grid, new reward values |
| **Modify** | `config/base/channel.yml` | mW units, add heights |
| **Modify** | `config/base/rl.yml` | Paper-aligned hyperparams |
| **Modify** | `config/base/map.yml` | 120x60, heights |
| **Modify** | `config/yml_config.py` | Load new params (heights, power levels) |

---

### Task 1: Update Config Files to Paper Parameters

**Files:**
- Modify: `config/base/map.yml`
- Modify: `config/base/channel.yml`
- Modify: `config/base/rl.yml`
- Modify: `config/base/env.yml`

- [ ] **Step 1: Update map.yml**

```yaml
description: "地图尺寸与禁区生成相关参数"

map_size:
  value: [120, 60]
  description: "地图尺寸 (rows, cols)，对应物理 48m x 24m，grid_size=0.4m"

antenna_position:
  value: [60, 30]
  description: "天线位置 (x, y) 网格坐标，物理坐标 (24m, 12m)"

number_of_robots:
  value: 4
  description: "机器人/智能体数量"

num_forbidden_squares:
  value: 5
  description: "禁区（方形）数量"

square_size_range:
  value: [7, 12]
  description: "方形禁区边长范围 (min, max)，原[3,5]按2.5倍缩放到新网格"

# 高度参数（新增）
h_AP:
  value: 2.0
  description: "接入点高度 (m)"

h_robot:
  value: 1.5
  description: "机器人接收天线高度 (m)"

h_block:
  value: 3.0
  description: "障碍物高度 (m)"
```

- [ ] **Step 2: Update channel.yml**

```yaml
description: "通信信道相关参数"

carrier_frequency:
  value: 3.5
  description: "载波频率 (GHz)"

sigma_rayleigh:
  value: 1.2
  description: "Rayleigh 分布的 sigma"

number_of_antenna:
  value: 128
  description: "天线数量 N_t"

power_AWGN:
  value: -143.0
  description: "AWGN功率谱密度 (dBm/Hz)"

channel_bandwidth:
  value: 1.0e+7
  description: "信道带宽 (Hz)，10 MHz"

channel_block_length:
  value: 256
  description: "信道块长度 N（论文参数表）"

packet_size:
  value: 16
  description: "数据包大小 D (bits)（论文参数表）"

P_sum:
  value: 100.0
  description: "总发射功率 (mW)，论文默认 100mW"

P_min_diff:
  value: 5.0
  description: "SIC最小功率差 (mW)"

num_power_levels:
  value: 3
  description: "可用功率数 p，功率空间大小，论文默认 p=3"
```

- [ ] **Step 3: Update rl.yml**

```yaml
description: "DQN / MADQN 训练与测试参数（对齐论文参数表）"

algo:
  value: "madqn"
  description: "算法选择：dqn 或 madqn"

lr:
  value: 1.0e-4
  description: "学习率（论文 0.0001）"

gamma:
  value: 0.9
  description: "折扣因子（论文 0.9）"

epsilon:
  value: 0.1
  description: "探索率（论文 0.1）"

epsilon_min:
  value: 0.1
  description: "最小探索率"

epsilon_decay:
  value: 1.0
  description: "探索率衰减系数（论文用固定 epsilon，设为 1.0 即不衰减）"

num_episodes:
  value: 200
  description: "训练 episode 数（论文 200 轮）"

episode_length:
  value: 5000
  description: "每个 episode 最大步数"

batch_size:
  value: 128
  description: "采样批次（论文 D_0=128）"

mini_batch_size:
  value: 128
  description: "小批量采样大小"

hidden_dim:
  value: 128
  description: "Q-网络隐藏层维度"

update_freq:
  value: 100
  description: "目标网络更新频率（论文 N_0=100）"

replay_buffer_size:
  value: 50000
  description: "经验池大小（论文 50000）"

test_max_steps:
  value: 500
  description: "测试时最大步数"

model_dir:
  value: "models"
  description: "模型保存/加载目录"
```

- [ ] **Step 4: Update env.yml**

```yaml
description: "环境与奖励参数（对齐论文）"

grid_size:
  value: 0.4
  description: "网格物理尺寸 (m)"

action_directions:
  value: [[0, 1], [1, 0], [0, -1], [-1, 0]]
  description: "移动方向：上右下左（不含停留，论文4方向）"

reward_goal:
  value: 10.0
  description: "到达终点奖励 R_goal（论文 10）"

reward_closer:
  value: -0.8
  description: "靠近终点奖励（论文 phi=0.8，靠近得 -phi，注意论文此处为负号表示'减少距离的代价'）"

reward_farther:
  value: 0.1
  description: "远离终点惩罚（论文 p=0.1）"

reward_same:
  value: 0.0
  description: "距离不变奖励"

reward_forbidden:
  value: -5.0
  description: "进入禁区惩罚"

omega:
  value: 0.004
  description: "通信奖励权重（论文默认 omega=0.004）"
```

- [ ] **Step 5: Commit**

```bash
git add config/base/map.yml config/base/channel.yml config/base/rl.yml config/base/env.yml
git commit -m "refactor: align all config parameters with reference paper"
```

---

### Task 2: Create Precomputed Radio Map Module

**Files:**
- Create: `communication/precompute.py`

- [ ] **Step 1: Create `communication/precompute.py`**

```python
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
            # area 格式: (row, col, width, height) 网格坐标
            if len(area) == 4:
                r, c, w, h = area
            elif len(area) == 2 and isinstance(area[0], (list, tuple, np.ndarray)):
                (r, c), (w, h) = area[0], (area[1][0] - area[0][0], area[1][1] - area[0][1])
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
            # 处理 dir_x == 0 和 dir_y == 0 的情况
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
            # (t 在 [0,1] 表示在起点和终点之间)
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
```

- [ ] **Step 2: Verify precompute works**

```bash
cd D:/MyProject/Python_Project/Graduation_Project
python -c "
from communication.precompute import PrecomputedRadioMap
import numpy as np
rm = PrecomputedRadioMap(
    map_size=(120, 60), grid_size=0.4,
    antenna_position=(60, 30), h_AP=2.0, h_robot=1.5, h_block=3.0,
    n_antenna=128, carrier_freq_ghz=3.5,
    forbidden_areas=[(10,10,8,8), (50,20,10,10)],
)
print('distances shape:', rm.distances.shape)
print('steering_vectors shape:', rm.steering_vectors.shape)
print('cache size:', os.path.getsize(rm._cache_path) / 1e6, 'MB')
H = rm.get_channel_vectors([(30, 15), (60, 30), (90, 45)])
print('H shape:', H.shape, 'dtype:', H.dtype)
# 第二次加载应走缓存
rm2 = PrecomputedRadioMap(
    map_size=(120, 60), grid_size=0.4,
    antenna_position=(60, 30), h_AP=2.0, h_robot=1.5, h_block=3.0,
    n_antenna=128, carrier_freq_ghz=3.5,
    forbidden_areas=[(10,10,8,8), (50,20,10,10)],
)
print('Cache hit test passed')
"
```

Expected: shapes (120,60), (120,60,128), H shape (3,128), cache ~14MB, cache hit on second load.

- [ ] **Step 3: Commit**

```bash
git add communication/precompute.py
git commit -m "feat: add precomputed radio map with npz caching"
```

---

### Task 3: Rewrite communication/SIC.py with Binary Power Control

**Files:**
- Rewrite: `communication/SIC.py`

- [ ] **Step 1: Rewrite SIC.py**

```python
"""
NOMA SIC 解码、二进制功率控制、SINR/BER 计算。
对齐论文：功率作为动作空间的一部分，每簇两用户按强弱分配不同功率等级。
"""
import numpy as np
from scipy.special import erfc
from scipy.stats import norm


def get_power_levels(P_max_per_cluster, num_levels):
    """
    生成二进制功率控制的功率等级列表（论文 3.2 节）。

    Args:
        P_max_per_cluster: 每簇最大功率 (mW)
        num_levels: 可用功率数 p

    Returns:
        strong_powers: 较强用户的功率等级列表，长度 p
                       P_max/2^p, ..., P_max/2^{2p} (从大到小排列，但都较小)
        weak_powers:   较弱用户的功率等级列表，长度 p
                       P_max/2, ..., P_max/2^{p} (从大到小排列，都较大)
    """
    p = num_levels
    # 较强机器人（信道好）分配较少功率
    strong_powers = np.array([P_max_per_cluster / (2 ** i) for i in range(p, 2 * p)])
    # 较弱机器人（信道差）分配较多功率
    weak_powers = np.array([P_max_per_cluster / (2 ** i) for i in range(1, p + 1)])
    return strong_powers, weak_powers


def compute_sinr(H_precoded, powers_strong, powers_weak, noise_power):
    """
    计算 SIC 后的 SINR（论文式 2-12, 2-13）。

    Args:
        H_precoded: (M, 2) array, |h_{m,i} * w_m|^2，M 簇各 2 用户的等效信道增益
        powers_strong: (M,) array, 每簇强用户分配功率 (mW)
        powers_weak: (M,) array, 每簇弱用户分配功率 (mW)
        noise_power: 噪声功率 (mW)

    Returns:
        sinr_strong: (M,) 强用户 SINR (SIC 后无簇内干扰)
        sinr_weak: (M,) 弱用户 SINR (含簇内干扰)
    """
    g_strong = H_precoded[:, 0]  # |h_{m,1} w_m|^2
    g_weak = H_precoded[:, 1]    # |h_{m,2} w_m|^2

    # 强用户 SIC 后 (式 2-12)
    sinr_strong = (powers_strong * g_strong) / noise_power

    # 弱用户 (式 2-13)
    sinr_weak = (powers_weak * g_weak) / (powers_strong * g_weak + noise_power)

    return sinr_strong, sinr_weak


def compute_ber(sinr, N, D):
    """
    有限块长下的解码错误概率（论文式 2-14 ~ 2-16）。

    Args:
        sinr: SINR 值 (线性，非 dB)
        N: 信道块长度
        D: 数据包大小 (bits)

    Returns:
        ber: 误码率 epsilon
    """
    sinr = np.maximum(sinr, 1e-10)  # 避免 log(0)

    # 信道色散 V (式 2-16)
    V = 1.0 - (1.0 + sinr) ** (-2)
    V = np.maximum(V, 1e-10)

    # 编码率
    rate = D / N

    # Q 函数参数 (式 2-14)
    capacity = np.log2(1.0 + sinr)
    xi = np.log(2) * np.sqrt(N / V) * (capacity - rate)

    # Q 函数: Q(x) = 0.5 * erfc(x / sqrt(2))
    ber = 0.5 * erfc(xi / np.sqrt(2))

    # 裁剪到 [1e-20, 1.0]
    ber = np.clip(ber, 1e-20, 1.0)

    return ber


def ber_to_reward(ber):
    """
    BER 转奖励（论文式 3-2）: R_rate = -log10(epsilon)。
    BER 越小，奖励越大（正值）。

    Args:
        ber: 误码率 array

    Returns:
        reward: -log10(ber)
    """
    ber = np.clip(ber, 1e-20, 1.0)
    return -np.log10(ber)
```

- [ ] **Step 2: Commit**

```bash
git add communication/SIC.py
git commit -m "refactor: rewrite SIC with binary power control aligned to paper"
```

---

### Task 4: Rewrite communication/ber_reward.py

**Files:**
- Rewrite: `communication/ber_reward.py`

- [ ] **Step 1: Rewrite ber_reward.py**

```python
"""
BER 奖励计算入口：整合预计算表、信道向量、分簇、预编码、SINR/BER。
env.step() 调用此模块的 compute_ber_rewards()。
"""
import numpy as np
from communication.precompute import PrecomputedRadioMap
from communication.SIC import compute_sinr, compute_ber, ber_to_reward, get_power_levels
from communication.diagonalization_precoding import matrix_cal


def cluster_agents(H, K):
    """
    NOMA 分簇：按信道增益 |h_k|^2 降序排列，第 m 名与第 K/2+m 名配对（论文 2.3.1）。

    Args:
        H: (K, N_t) complex，信道向量矩阵
        K: agent 数量

    Returns:
        clusters: list of (strong_idx, weak_idx)，每簇包含强弱用户的原始索引
        sorted_indices: 按信道增益降序排列的索引
    """
    gains = np.sum(np.abs(H) ** 2, axis=1)  # (K,) 信道增益
    sorted_indices = np.argsort(-gains)  # 降序

    M = K // 2
    clusters = []
    for m in range(M):
        strong_idx = sorted_indices[m]       # 信道好的
        weak_idx = sorted_indices[M + m]     # 信道差的
        clusters.append((strong_idx, weak_idx))

    return clusters, sorted_indices


def compute_ber_rewards(
    radio_map: PrecomputedRadioMap,
    positions,
    power_actions,
    P_sum,
    num_power_levels,
    N,
    D,
    noise_power,
    rng=None,
):
    """
    完整 BER 奖励计算流程。

    Args:
        radio_map: PrecomputedRadioMap 实例
        positions: (K, 2) int array，agent 网格坐标
        power_actions: (K,) int array，每个 agent 选择的功率等级索引
        P_sum: 总发射功率 (mW)
        num_power_levels: 可用功率数 p
        N: 信道块长度
        D: 数据包大小
        noise_power: 噪声功率 (mW)
        rng: numpy random generator

    Returns:
        dict with:
            ber: (K,) 每个 agent 的 BER
            sinr: (K,) 每个 agent 的 SINR
            reward: (K,) 每个 agent 的通信奖励 R_rate
    """
    positions = np.array(positions, dtype=int)
    K = len(positions)

    # 1. 获取信道向量（查预计算表 + Rayleigh 衰落）
    H = radio_map.get_channel_vectors(positions, rng=rng)

    # 2. 处理 K=1 的特殊情况
    if K == 1:
        gain = np.sum(np.abs(H[0]) ** 2)
        P_max_cluster = P_sum
        strong_powers, _ = get_power_levels(P_max_cluster, num_power_levels)
        p_idx = min(power_actions[0], len(strong_powers) - 1)
        power = strong_powers[p_idx]
        sinr_val = (power * gain) / noise_power
        ber_val = compute_ber(np.array([sinr_val]), N, D)
        reward_val = ber_to_reward(ber_val)
        return {
            "ber": ber_val,
            "sinr": np.array([sinr_val]),
            "reward": reward_val,
        }

    # 3. 分簇
    clusters, sorted_indices = cluster_agents(H, K)
    M = len(clusters)
    P_max_per_cluster = P_sum / M

    # 功率等级表
    strong_powers_table, weak_powers_table = get_power_levels(P_max_per_cluster, num_power_levels)

    # 4. BD 预编码
    # 构建每簇的信道矩阵
    H_clusters = []
    for s_idx, w_idx in clusters:
        H_clusters.append(np.vstack([H[s_idx:s_idx+1], H[w_idx:w_idx+1]]))  # (2, N_t)

    W = matrix_cal(H_clusters)  # list of (N_t, 1) precoding vectors

    # 5. 计算等效信道增益和 SINR
    ber_all = np.zeros(K)
    sinr_all = np.zeros(K)

    for m, (s_idx, w_idx) in enumerate(clusters):
        w_m = W[m]  # (N_t, 1)

        # 等效信道增益 |h * w|^2
        g_strong = np.abs(H[s_idx] @ w_m) ** 2
        g_weak = np.abs(H[w_idx] @ w_m) ** 2

        # 根据 agent 的功率动作选择功率
        p_s_idx = min(power_actions[s_idx], len(strong_powers_table) - 1)
        p_w_idx = min(power_actions[w_idx], len(weak_powers_table) - 1)
        p_strong = strong_powers_table[p_s_idx]
        p_weak = weak_powers_table[p_w_idx]

        # SINR
        sinr_s = float((p_strong * g_strong) / noise_power)
        sinr_w = float((p_weak * g_weak) / (p_strong * g_weak + noise_power))

        sinr_all[s_idx] = sinr_s
        sinr_all[w_idx] = sinr_w

        # BER
        ber_all[s_idx] = float(compute_ber(np.array([sinr_s]), N, D)[0])
        ber_all[w_idx] = float(compute_ber(np.array([sinr_w]), N, D)[0])

    # 处理 K 为奇数时落单的 agent
    if K % 2 == 1:
        last_idx = sorted_indices[-1]
        gain = np.sum(np.abs(H[last_idx]) ** 2)
        p_idx = min(power_actions[last_idx], len(strong_powers_table) - 1)
        power = strong_powers_table[p_idx]
        sinr_val = (power * gain) / noise_power
        sinr_all[last_idx] = sinr_val
        ber_all[last_idx] = float(compute_ber(np.array([sinr_val]), N, D)[0])

    reward_all = ber_to_reward(ber_all)

    return {
        "ber": ber_all,
        "sinr": sinr_all,
        "reward": reward_all,
    }
```

- [ ] **Step 2: Commit**

```bash
git add communication/ber_reward.py
git commit -m "refactor: rewrite ber_reward with precomputed radio map and power actions"
```

---

### Task 5: Rewrite env/env.py

**Files:**
- Rewrite: `env/env.py`

This is the largest change. The new env has:
- 120x60 grid (48m x 24m)
- Action space = 4 directions x p power levels
- Paper-aligned reward function
- Uses PrecomputedRadioMap for fast BER computation

- [ ] **Step 1: Rewrite env.py**

```python
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
                channel_block_length, packet_size, noise_power_mw,
                los_nlos_grid (optional)
        """
        # 地图
        self.map_size = tuple(int(x) for x in config["map_size"])
        self.rows, self.cols = self.map_size
        self.grid_size = config["grid_size"]
        self.n_states = self.rows * self.cols

        # Agent 起终点
        self.start_states = [tuple(s) for s in config["start_states"]]
        self.target_states = [tuple(s) for s in config["target_states"]]
        self.num_agents = len(self.start_states)

        # 禁区 -> 占用网格集合
        self.forbidden_areas_raw = config["forbidden_areas"]
        self.forbidden_set = self._build_forbidden_set(self.forbidden_areas_raw)

        # 动作空间
        self.directions = [tuple(d) for d in config["action_directions"]]
        self.n_dirs = len(self.directions)
        self.n_powers = config["num_power_levels"]
        self.n_actions = self.n_dirs * self.n_powers

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

    def _build_forbidden_set(self, areas):
        """将禁区列表转为网格坐标集合。"""
        forbidden = set()
        for area in areas:
            if len(area) == 4:
                r, c, w, h = area
                for dr in range(int(w)):
                    for dc in range(int(h)):
                        forbidden.add((int(r + dr), int(c + dc)))
            elif len(area) == 2:
                # [(r1,c1), (r2,c2)] 格式
                (r1, c1), (r2, c2) = area
                for r in range(int(r1), int(r2)):
                    for c in range(int(c1), int(c2)):
                        forbidden.add((r, c))
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
```

- [ ] **Step 2: Commit**

```bash
git add env/env.py
git commit -m "refactor: rewrite env with 120x60 grid, compound actions, paper-aligned rewards"
```

---

### Task 6: Update Config Loader (yml_config.py)

**Files:**
- Modify: `config/yml_config.py`

- [ ] **Step 1: Update get_channel_config() to load new params**

In `config/yml_config.py`, update the `get_channel_config()` function to include new parameters (heights, P_sum, num_power_levels) and compute noise_power_mw:

```python
def get_channel_config() -> _ChannelArgs:
    _channel_yml = _load_base_yml("channel")
    _map_yml = _load_base_yml("map")
    ap = _get_yml_value(_channel_yml, "antenna_position", None)
    if ap is None:
        ap = _get_yml_value(_map_yml, "antenna_position", [60, 30])
    ap = tuple(ap) if isinstance(ap, list) else ap

    power_awgn_dbm_hz = float(_get_yml_value(_channel_yml, "power_AWGN", -143.0))
    bw = float(_get_yml_value(_channel_yml, "channel_bandwidth", 1.0e7))
    # 噪声功率 (mW): N0(dBm/Hz) + 10*log10(BW) -> dBm -> mW
    noise_dbm = power_awgn_dbm_hz + 10 * np.log10(bw)
    noise_power_mw = 10 ** (noise_dbm / 10)  # dBm -> mW

    return _ChannelArgs({
        "carrier_frequency": float(_get_yml_value(_channel_yml, "carrier_frequency", 3.5)),
        "sigma_rayleigh": float(_get_yml_value(_channel_yml, "sigma_rayleigh", 1.2)),
        "number_of_antenna": int(_get_yml_value(_channel_yml, "number_of_antenna", 128)),
        "antenna_position": ap,
        "power_AWGN": power_awgn_dbm_hz,
        "channel_bandwidth": bw,
        "channel_block_length": int(_get_yml_value(_channel_yml, "channel_block_length", 256)),
        "packet_size": int(_get_yml_value(_channel_yml, "packet_size", 16)),
        "P_sum": float(_get_yml_value(_channel_yml, "P_sum", 100.0)),
        "P_min_diff": float(_get_yml_value(_channel_yml, "P_min_diff", 5.0)),
        "num_power_levels": int(_get_yml_value(_channel_yml, "num_power_levels", 3)),
        "noise_power_mw": noise_power_mw,
    })
```

- [ ] **Step 2: Update get_env_config() to include new params**

Add height params and channel params to the env config dict returned by `get_env_config()`:

```python
    # At end of get_env_config(), before the return statement, add:
    _channel_cfg = get_channel_config()
    _map_yml_heights = _load_base_yml("map")

    # ... existing code ...

    return {
        # existing keys...
        "map_size": map_size,
        "grid_size": float(_get_yml_value(_env_yml, "grid_size", 0.4)),
        "start_states": scenario["start_states"],
        "target_states": scenario["target_states"],
        "forbidden_areas": scenario["forbidden_areas"],
        "action_directions": _get_yml_value(_env_yml, "action_directions", [[0,1],[1,0],[0,-1],[-1,0]]),
        "reward_goal": float(_get_yml_value(_env_yml, "reward_goal", 10)),
        "reward_forbidden": float(_get_yml_value(_env_yml, "reward_forbidden", -5)),
        "reward_closer": float(_get_yml_value(_env_yml, "reward_closer", -0.8)),
        "reward_farther": float(_get_yml_value(_env_yml, "reward_farther", 0.1)),
        "reward_same": float(_get_yml_value(_env_yml, "reward_same", 0.0)),
        "omega": float(_get_yml_value(_env_yml, "omega", 0.004)),
        "antenna_position": antenna_position,
        "los_nlos_grid": scenario["los_nlos_grid"],
        # new keys
        "h_AP": float(_get_yml_value(_map_yml_heights, "h_AP", 2.0)),
        "h_robot": float(_get_yml_value(_map_yml_heights, "h_robot", 1.5)),
        "h_block": float(_get_yml_value(_map_yml_heights, "h_block", 3.0)),
        "n_antenna": _channel_cfg.number_of_antenna,
        "carrier_freq_ghz": _channel_cfg.carrier_frequency,
        "sigma_rayleigh": _channel_cfg.sigma_rayleigh,
        "P_sum": _channel_cfg.P_sum,
        "P_min_diff": _channel_cfg.P_min_diff,
        "num_power_levels": _channel_cfg.num_power_levels,
        "channel_block_length": _channel_cfg.channel_block_length,
        "packet_size": _channel_cfg.packet_size,
        "noise_power_mw": _channel_cfg.noise_power_mw,
    }
```

- [ ] **Step 3: Update get_rl_config defaults**

```python
_RL_DEFAULTS = dict(
    algo="madqn", lr=1e-4, gamma=0.9,
    epsilon=0.1, epsilon_min=0.1, epsilon_decay=1.0,
    num_episodes=200, episode_length=5000,
    batch_size=128, mini_batch_size=128, hidden_dim=128, update_freq=100,
    replay_buffer_size=50000,
    test_max_steps=500, model_dir="models",
)
```

- [ ] **Step 4: Commit**

```bash
git add config/yml_config.py
git commit -m "refactor: update config loader for new params (heights, power levels, noise)"
```

---

### Task 7: Update Q-Network for Compound Action Space

**Files:**
- Modify: `rl_algorithms/net/qnet.py`

- [ ] **Step 1: Update Qnet to accept n_actions parameter**

The current Qnet hardcodes `action_dim=5`. Change to accept `n_actions` as parameter (default = 4 dirs x 3 powers = 12).

In `qnet.py`, change the `__init__` signature and the output layer:

```python
class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_states=None):
        super().__init__()
        if num_states is None:
            num_states = state_dim
        embed_dim = hidden_dim

        # 状态 embedding 分支
        self.state_embedding = torch.nn.Embedding(num_states, embed_dim)
        self.state_fc = torch.nn.Linear(embed_dim, hidden_dim)

        # 目标 embedding 分支（共享 embedding）
        self.target_fc = torch.nn.Linear(embed_dim, hidden_dim)

        # 相对位置分支
        self.rel_fc1 = torch.nn.Linear(8, hidden_dim)
        self.rel_fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        # 融合头
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),  # action_dim = n_dirs * n_powers
        )

        self.num_states = num_states
        self.map_cols = None  # 需要外部设置用于相对位置计算
```

The rest of the forward pass stays the same. The key change is `action_dim` is now passed in (12 instead of 5).

- [ ] **Step 2: Commit**

```bash
git add rl_algorithms/net/qnet.py
git commit -m "refactor: qnet accepts configurable action_dim for compound actions"
```

---

### Task 8: Update MADQN Agent for Compound Actions

**Files:**
- Modify: `rl_algorithms/structure/madqn.py`

- [ ] **Step 1: Update MADQN to handle compound action space**

Key changes:
1. `__init__`: accept `n_actions` (= n_dirs * n_powers), `n_dirs`, `n_powers`
2. `take_action`: epsilon-greedy over compound actions, collision avoidance checks direction component only
3. `update`: no change needed (standard DQN update, action indices still work)

```python
class MADQN:
    def __init__(self, num_agents, state_dim, hidden_dim, n_actions,
                 n_dirs, n_powers, lr, gamma, epsilon, epsilon_min,
                 epsilon_decay, device, map_cols,
                 replay_buffer_size=50000):
        self.num_agents = num_agents
        self.n_actions = n_actions
        self.n_dirs = n_dirs
        self.n_powers = n_powers
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.map_cols = map_cols

        num_states = state_dim  # = rows * cols = 7200

        self.q_nets = []
        self.target_nets = []
        self.optimizers = []
        self.buffers = []

        for _ in range(num_agents):
            q = Qnet(state_dim, hidden_dim, n_actions, num_states=num_states).to(device)
            t = Qnet(state_dim, hidden_dim, n_actions, num_states=num_states).to(device)
            t.load_state_dict(q.state_dict())
            q.map_cols = map_cols
            t.map_cols = map_cols
            self.q_nets.append(q)
            self.target_nets.append(t)
            self.optimizers.append(torch.optim.Adam(q.parameters(), lr=lr))
            self.buffers.append(ReplayBuffer(replay_buffer_size))

    def decode_action(self, action):
        """compound action -> (dir_idx, power_idx)"""
        return action // self.n_powers, action % self.n_powers

    def take_action(self, states, targets, env_positions, forbidden_set):
        """
        Epsilon-greedy with collision avoidance on direction component.

        Args:
            states: list of state_index (int)
            targets: list of target_index (int)
            env_positions: current (num_agents, 2) positions
            forbidden_set: set of (r,c) forbidden cells

        Returns:
            actions: list of compound action indices
        """
        actions = []
        occupied = set()  # 已选定的下一步位置

        directions = [(0,1),(1,0),(0,-1),(-1,0)]

        for i in range(self.num_agents):
            state = torch.tensor([states[i]], dtype=torch.long, device=self.device)
            target = torch.tensor([targets[i]], dtype=torch.long, device=self.device)

            if np.random.random() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                with torch.no_grad():
                    q_values = self.q_nets[i](state, target)
                action = q_values.argmax(dim=1).item()

            # 检查方向是否导致冲突
            dir_idx, power_idx = self.decode_action(action)
            dr, dc = directions[dir_idx]
            new_r = int(env_positions[i, 0] + dr)
            new_c = int(env_positions[i, 1] + dc)
            rows, cols = env_positions.shape[0], self.map_cols  # rough bounds

            # 如果越界或冲突，尝试其他动作
            if (new_r, new_c) in occupied or (new_r, new_c) in forbidden_set:
                # 按 Q 值排序尝试其他动作
                with torch.no_grad():
                    q_values = self.q_nets[i](state, target)
                sorted_actions = q_values.argsort(dim=1, descending=True).squeeze().tolist()
                for alt_action in sorted_actions:
                    d_idx, p_idx = self.decode_action(alt_action)
                    dr2, dc2 = directions[d_idx]
                    nr2 = int(env_positions[i, 0] + dr2)
                    nc2 = int(env_positions[i, 1] + dc2)
                    if (nr2, nc2) not in occupied and (nr2, nc2) not in forbidden_set:
                        action = alt_action
                        new_r, new_c = nr2, nc2
                        break

            occupied.add((new_r, new_c))
            actions.append(action)

        return actions
```

- [ ] **Step 2: Commit**

```bash
git add rl_algorithms/structure/madqn.py
git commit -m "refactor: MADQN handles compound direction+power actions"
```

---

### Task 9: Update DQN Agent Similarly

**Files:**
- Modify: `rl_algorithms/structure/dqn.py`

- [ ] **Step 1: Update DQN with same compound action changes**

Same pattern as MADQN but for single agent: accept `n_actions`, `n_dirs`, `n_powers` in init, Qnet output dim = n_actions.

- [ ] **Step 2: Commit**

```bash
git add rl_algorithms/structure/dqn.py
git commit -m "refactor: DQN handles compound direction+power actions"
```

---

### Task 10: Update Training Loops

**Files:**
- Modify: `rl_algorithms/train/train_madqn.py`
- Modify: `rl_algorithms/train/train_dqn.py`
- Modify: `rl_algorithms/train/run.py`

- [ ] **Step 1: Update train_madqn.py**

Key changes:
- Remove `iteration` outer loop (paper uses flat 200 episodes)
- Pass `forbidden_set` and `positions` to `take_action`
- Collect BER from `info` dict
- Use new env API (compound actions, new reward structure)

```python
def train_madqn(env, agent, num_episodes, episode_length, logger=None):
    return_list = []
    ber_list = []

    for ep in range(num_episodes):
        states = env.reset()
        targets = [env.pos_to_index(*t) for t in env.target_states]
        ep_return = np.zeros(env.num_agents)
        ep_ber = []

        for step in range(episode_length):
            actions = agent.take_action(
                states, targets, env.positions, env.forbidden_set
            )
            next_states, rewards, dones, info = env.step(actions)

            # 存经验
            for i in range(env.num_agents):
                if not env.done_flags[i] or dones[i]:  # 刚到达的也存
                    agent.buffers[i].add(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    ep_return[i] += rewards[i]

            # 更新网络
            for i in range(env.num_agents):
                if len(agent.buffers[i]) >= agent.buffers[i].capacity * 0.1:
                    agent.update(i, agent.buffers[i].sample(agent.batch_size))

            # 更新目标网络
            if step % agent.update_freq == 0:
                for i in range(env.num_agents):
                    agent.target_nets[i].load_state_dict(agent.q_nets[i].state_dict())

            ep_ber.append(info["ber"].mean())
            states = next_states

            if env.all_done:
                break

        return_list.append(ep_return.sum())
        ber_list.append(np.mean(ep_ber) if ep_ber else 0)

        if logger:
            logger.debug(f"Ep {ep+1}/{num_episodes} | Return={ep_return.sum():.2f} | "
                        f"AgentReturns=[{', '.join(f'A{i+1}:{r:.2f}' for i,r in enumerate(ep_return))}] | "
                        f"AvgBER={ber_list[-1]:.6f}")

    return {"return_list": return_list, "ber_list": ber_list}
```

- [ ] **Step 2: Update run.py to construct env and agent with new params**

In `rl_algorithms/train/run.py`, update the `train()` function to:
- Build `MultiRobotEnv` from `get_env_config()`
- Build MADQN with `n_actions = n_dirs * n_powers`
- Pass `replay_buffer_size` from rl config

- [ ] **Step 3: Similarly update train_dqn.py**

- [ ] **Step 4: Commit**

```bash
git add rl_algorithms/train/train_madqn.py rl_algorithms/train/train_dqn.py rl_algorithms/train/run.py
git commit -m "refactor: training loops use new env API with compound actions"
```

---

### Task 11: Update Test/Visualization

**Files:**
- Modify: `rl_algorithms/test/run.py`
- Modify: `rl_algorithms/test/test_madqn.py`
- Modify: `rl_algorithms/test/test_dqn.py`

- [ ] **Step 1: Update test run.py**

Adapt test runner to new env API:
- Construct env same way as training
- Use compound actions from agent
- Collect frame data for GIF generation via `env.save_gif()`

- [ ] **Step 2: Update test_madqn.py and test_dqn.py**

Remove hardcoded paths, use config-driven model paths.

- [ ] **Step 3: Commit**

```bash
git add rl_algorithms/test/run.py rl_algorithms/test/test_madqn.py rl_algorithms/test/test_dqn.py
git commit -m "refactor: test scripts use new env API"
```

---

### Task 12: Update ReplayBuffer with Configurable Size

**Files:**
- Modify: `rl_algorithms/utils/replaybuffer.py`

- [ ] **Step 1: Add capacity parameter**

```python
import collections
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
```

- [ ] **Step 2: Commit**

```bash
git add rl_algorithms/utils/replaybuffer.py
git commit -m "refactor: replay buffer with configurable capacity (default 50000)"
```

---

### Task 13: Regenerate Dynamic Config for 120x60 Grid

**Files:**
- Modify: `config/generator/main.py` (if needed for new grid size)

- [ ] **Step 1: Delete old dynamic config and regenerate**

```bash
rm -f config/dynamic/*.yml
python -c "
from config.yml_config import get_env_config
cfg = get_env_config()
print('map_size:', cfg['map_size'])
print('num_agents:', len(cfg['start_states']))
print('forbidden_areas count:', len(cfg['forbidden_areas']))
print('Generated successfully')
"
```

Expected: map_size (120, 60), 4 agents, 5 forbidden areas.

- [ ] **Step 2: Commit**

```bash
git add config/dynamic/
git commit -m "refactor: regenerate dynamic config for 120x60 grid"
```

---

### Task 14: Smoke Test - Full Training Run

- [ ] **Step 1: Run a short training to verify everything connects**

```bash
python main.py --model madqn --mode train
```

If training starts and runs for at least a few episodes without errors, the refactor is functionally complete.

- [ ] **Step 2: Run a test with a saved model**

```bash
python main.py --model madqn --mode test
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "refactor: complete paper-aligned refactoring (120x60, compound actions, precomputed radio map)"
```

---

## Summary of Performance Impact

| Operation | Before (per step) | After (per step) |
|-----------|-------------------|------------------|
| Distance calculation | K * O(1) | **0** (cached) |
| AoA calculation | K * O(1) | **0** (cached) |
| LOS/NLOS check | K * O(obstacles) | **0** (cached) |
| Path loss | K * O(1) | **0** (cached) |
| Steering vector | K * O(N_t) | **0** (cached) |
| Rayleigh fading | K * O(N_t) | K * O(N_t) |
| Clustering | O(K log K) | O(K log K) |
| BD Precoding (SVD) | O(M * N_t^2) | O(M * N_t^2) |
| SINR/BER | O(K) | O(K) |

Cache cold start: ~2 seconds for 120x60x128. Subsequent runs: ~50ms (npz load).
