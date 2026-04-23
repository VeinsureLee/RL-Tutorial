# config 模块说明

本模块是整个项目的配置中枢。对外只暴露一个入口 `config/yml_config.py`，内部维护两层数据：

1. **静态标量层（`config/base/*.yml`）**：人工编辑的超参（地图尺寸、奖励权重、信道参数、RL 超参、随机种子）。结构统一为 `{ value: ..., description: ... }`，通过 `_get_yml_value` 读取。
2. **动态场景层（`config/dynamic/*.npz`）**：由生成器/预计算器自动落盘，包括随机场景 (`scenario.npz`) 与无线电地图缓存 (`radio_map_cache.npz`)。YAML 参数改变时自动失效并重建。

**代码与数据分离**——重构后：

- **代码**在 `src/config/`（Python 包 `config`）：`yml_config.py` + `generator/`
- **数据**在项目根 `config/`：`base/*.yml`（人工编辑）+ `dynamic/*.npz`（运行时生成）

```
<repo_root>/
├── config/                        # ← 数据层（人可见、可编辑）
│   ├── base/
│   │   ├── map.yml                # 地图、AP、机器人数量、障碍生成
│   │   ├── env.yml                # 网格、动作集、奖励权重
│   │   ├── channel.yml            # 信道/天线/发射功率
│   │   ├── rl.yml                 # DQN / MADQN 超参
│   │   └── random_seed.yml        # 全局随机种子
│   └── dynamic/                   # 自动生成
│       ├── scenario.npz           # 随机起终点 + 禁区
│       └── radio_map_cache.npz    # 预算好的距离/AoA/LOS/PL/steering
│
└── src/config/                    # ← 代码层
    ├── yml_config.py              # 对外唯一入口：合并两层配置
    └── generator/                 # 场景生成器
        ├── main.py                # save/load/get_or_create_scenario
        ├── forbidden_generator.py # 障碍物方框生成
        ├── states_generator.py    # 起终点随机采样
        └── environment_validation.py
```

**路径解析**：`utils.path_tool.get_abs_path("config/base/map.yml")` 始终返回项目根下的
`<repo_root>/config/base/map.yml`。代码里任何对 `"config/..."` 的相对引用都指向这份数据，
不会误指 `src/config/`。

---

## 1. `base/map.yml` — 地图与场景基础参数

| 参数 | 含义 | 默认值 | 备注 / 公式 |
| --- | --- | --- | --- |
| `map_size` | 栅格尺寸 `(rows, cols)` | `[120, 60]` | 物理尺寸 = `map_size × grid_size`，默认 `48m × 24m` |
| `antenna_position` | AP 在栅格中的坐标 `(x, y)` | `[60, 30]` | 默认在地图正中心；不能落在禁区内 |
| `number_of_robots` | 智能体数量 `K` | `1` | 影响 `scenario.npz` 中起终点个数；改动会自动触发场景重建 |
| `num_forbidden_squares` | 方形禁区数量 `M` | `5` | 生成器会保证不与 AP 重叠、互不重叠 |
| `square_size_range` | 禁区边长采样区间 `[lo, hi)` | `[3, 5]` | `np.random.randint(lo, hi)` 左闭右开 |
| `h_AP` | AP 天线高度 (m) | `2.0` | 参与 3D 距离计算 |
| `h_robot` | 机器人接收天线高度 (m) | `1.5` | 同上，3D 距离中 `dz = h_robot − h_AP` |
| `h_block` | 障碍物高度 (m) | `3.0` | 目前 LOS 判定按 2D AABB 进行，高度参数保留供后续 3D 扩展 |

**物理坐标与栅格坐标的转换：**
```
physical_coord (m) = (grid_index + 0.5) × grid_size
```
即每个栅格中心在物理空间的位置。

---

## 2. `base/env.yml` — 环境与奖励

| 参数 | 含义 | 默认值 | 公式 / 说明 |
| --- | --- | --- | --- |
| `grid_size` | 单格物理边长 (m) | `0.4` | 与 `map_size` 共同决定物理尺寸 |
| `action_directions` | 单智能体移动方向集合 | `[[0,1],[1,0],[0,-1],[-1,0],[0,0]]` | 5 个动作（右/下/左/上/停），与 3bede13 版本对齐 |
| `reward_goal` | 到达终点奖励 | `50.0` | 一次性给予，结束本 agent 的交互 |
| `reward_closer` | 朝目标走近的奖励 | `+1.0` | 对比 `‖pos_t − target‖ − ‖pos_{t−1} − target‖ < 0` |
| `reward_farther` | 远离目标的惩罚 | `−1.0` | 对应 `> 0` 情况 |
| `reward_same` | 距离不变的奖励 | `0.0` | 常用于 STAY 动作 |
| `reward_forbidden` | 撞禁区 / 越界惩罚 | `−5.0` | 动作被回滚，位置不变 |
| `reward_step` | 每步时间惩罚 | `−1.0` | 对所有仍在线的 agent 都扣，促使尽快到达 |
| `omega` | 通信奖励权重 `ω` | `1.0` | `comm_reward = ω · ΔBER_term` |
| `ber_reward_better` | BER 下降时的奖励 | `+1.0` | 当 `BER_t < BER_{t−1}` |
| `ber_reward_worse` | BER 上升时的惩罚 | `−1.0` | 当 `BER_t > BER_{t−1}` |

**总奖励分解**：
```
r_t = step_reward + approach_reward + comm_reward
    = reward_step + {closer|farther|same|forbidden|goal} + ω·{better|worse|0}
```

**注意**：BER 项是**增量 (delta)** 奖励，与上一步比较，而不是把绝对 BER 映射成分数。

---

## 3. `base/channel.yml` — 信道与 NOMA / BD / SIC 参数

| 参数 | 含义 | 默认值 | 公式 / 说明 |
| --- | --- | --- | --- |
| `carrier_frequency` | 载波频率 (GHz) | `3.5` | 波长 `λ = 3×10⁸ / (fc × 10⁹)`，天线间距 `d = λ/2` |
| `sigma_rayleigh` | Rayleigh 衰落 `σ` | `1.2` | 小尺度衰落 `h ~ CN(0, σ²)`，实/虚部各 `~ N(0, σ²/2)` |
| `number_of_antenna` | AP 天线数 `N_t` | `128` | ULA 阵列，阵列响应 `a_n(θ) = exp(−jπ·n·sinθ)/√N_t` |
| `antenna_position` | AP 坐标（可覆盖 `map.yml`） | 缺省→读 `map.yml` | 读入优先级：`channel.yml → map.yml → scenario.npz` |
| `power_AWGN` | AWGN 功率谱密度 `N₀` (dBm/Hz) | `−143.0` | |
| `channel_bandwidth` | 信道带宽 `B` (Hz) | `1.0e7` (10 MHz) | 用于噪声功率换算 |
| `channel_block_length` | 有限码长 `N` (符号) | `256` | 用于 URLLC 有限码长 BER 计算 |
| `packet_size` | 单包比特数 `D` | `16` | 码率 `R = D/N` |
| `P_sum` | 总发射功率 (mW) | `100.0` | |
| `P_min_diff` | SIC 最小功率差 (mW) | `5.0` | **注：当前 `yml_config.get_channel_config` 未读取此项**，仅保留在 YAML 中 |
| `num_power_levels` | 离散功率等级数 `p` | `1` | 动作空间 `n_actions = n_dirs × num_power_levels`。纯导航训练阶段设为 1（退化为无功率选择） |

### 派生量

- **噪声功率（mW）**：
  ```
  N_dBm  = N₀ + 10·log₁₀(B)       # 例：−143 + 70 = −73 dBm
  N_mW   = 10^(N_dBm / 10)          # 例：5.01e−8 mW
  ```
  代码位于 `yml_config.get_channel_config()`，字段名 `noise_power_mw`。

- **波长与阵列间距**：
  ```
  λ = 3e8 / (f_c · 1e9)             # m
  d = λ / 2                          # 阵元间距
  ```

- **路径损耗（precompute.py 中实现）**：
  ```
  PL_LOS  = 31.84 + 21.50·log₁₀(d₃d) + 19.00·log₁₀(fc)   (dB)
  PL_SL   = 33.00 + 25.50·log₁₀(d₃d) + 20.00·log₁₀(fc)   (dB)
  PL_NLOS = max(PL_SL, PL_LOS)
  PL      = PL_LOS  if LOS  else  PL_NLOS
  ```

- **大尺度衰落系数**：
  ```
  β = 10^(−PL / 20)
  ```

- **信道向量**（每步查表 + Rayleigh）：
  ```
  h_k = β_k · a(θ_k) · g_k,    g_k ~ CN(0, σ²)
  ```

---

## 4. `base/rl.yml` — DQN / MADQN 超参

| 参数 | 含义 | 默认值 | 备注 |
| --- | --- | --- | --- |
| `algo` | 算法选择 | `"madqn"` | `dqn` 或 `madqn`；CLI `--model` 可覆盖 |
| `lr` | 学习率 | `1e-4` | |
| `gamma` | 折扣因子 γ | `0.99` | 有效回望约 100 步 |
| `epsilon` | 初始 ε | `0.5` | 每次 iteration 开始会重置 |
| `epsilon_min` | ε 下限 | `0.05` | |
| `epsilon_decay` | 每 episode ε 衰减系数 | `0.95` | `ε ← max(ε·decay, ε_min)` |
| `num_iterations` | 外层迭代轮次 | `1` | |
| `num_episodes` | 每轮的 episode 数 | `50` | |
| `episode_length` | 每 episode 最大步数 | `5000` | |
| `batch_size` | 采样 minibatch 大小 | `64` | |
| `hidden_dim` | Q 网络隐藏层维度 | `128` | |
| `update_freq` | 目标网络同步频率（env step） | `10` | 每 N 步 hard-copy |
| `replay_buffer_size` | 回放池容量 | `50000` | |
| `train_interval` | 梯度更新频率（env step） | `1` | 每 N 步更新一次 |
| `test_max_steps` | 测试最大步数 | `500` | |
| `model_dir` | 权重落盘目录 | `"models"` | |

**ε 衰减公式**：
```
ε_{e+1} = max(ε_e · epsilon_decay, epsilon_min)
```

**覆盖优先级**：CLI 参数 > `rl.yml` > `_RL_DEFAULTS`（`yml_config.py`）。`get_rl_config(**overrides)` 中值为 `None` 的覆盖会被忽略。

---

## 5. `base/random_seed.yml`

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `random_seed` | 全局随机种子 | `42` |

影响面：`forbidden_generator`（禁区）、`states_generator`（起终点）。训练中的探索随机性由各 agent 自行管理。

---

## 6. `dynamic/scenario.npz` — 场景快照

由 `config/generator/main.py` 写入，键值如下：

| 键 | 形状 / dtype | 含义 |
| --- | --- | --- |
| `map_size` | `(2,) int32` | `(rows, cols)` |
| `num_agents` | `() int32` | 智能体数 |
| `antenna_position` | `(2,) int32` | AP 栅格坐标 |
| `start_states` | `(K, 2) int32` | 起点集合 |
| `target_states` | `(K, 2) int32` | 终点集合 |
| `forbidden_areas` | `(M, 3) int32` | 每行 `(row, col, size)`，方形禁区 |

**生成流程**（`get_or_create_scenario`）：

1. `forbidden_generator.generate_forbidden_areas` — 固定种子下随机放 `M` 个方形禁区，保证不与 AP 重叠、互不相交。
2. `states_generator.generate_states` — 不落在禁区或已占位上，起点集合与终点集合不重合。
3. `environment_validation.validate_environment_parameters` — 越界 / 重叠检查。
4. `np.savez_compressed` 写入 `scenario.npz`。

**自动重建触发条件**（`_ensure_scenario`）：
- 文件不存在
- `num_agents` 与 YAML 不一致
- `map_size` 与 YAML 不一致

重建时**种子固定为 `random_seed.yml:random_seed`**，因此只要 YAML 未改，场景可复现。

---

## 7. `dynamic/radio_map_cache.npz` — 预计算无线电地图

由 `communication/precompute.PrecomputedRadioMap` 写入。参数哈希命中即复用，否则重建。

| 键 | 形状 / dtype | 含义 |
| --- | --- | --- |
| `param_hash` | str | md5 指纹（见下） |
| `distances` | `(rows, cols) float64` | 每个栅格到 AP 的 **3D** 距离 (m)，下限 0.1 |
| `aoa` | `(rows, cols) float64` | 到达角 `θ = arctan2(Δy, Δx)` (rad) |
| `los_grid` | `(rows, cols) bool` | LOS 判定结果（AABB 射线相交） |
| `path_loss` | `(rows, cols) float64` | 路径损耗 PL (dB) |
| `beta` | `(rows, cols) float64` | `10^(−PL/20)` |
| `steering_vectors` | `(rows, cols, N_t) complex128` | `a_n(θ) = e^{−jπn·sinθ}/√N_t` |

**参数哈希**（`_compute_param_hash`）基于：`map_size, grid_size, ap_grid, ap_pos_m, h_robot, h_block, n_antenna, carrier_freq_ghz, forbidden_areas`。这些字段任何改变都会使缓存失效。

**LOS 判定**：对每个禁区做 slab-method AABB 射线相交测试。射线方向从栅格点指向 AP；`tmin ≤ tmax` 且 `tmin < 1`、`tmax > 0` 即视为被遮挡。

---

## 8. `yml_config.py` 对外接口

| 函数 | 返回 | 使用方 |
| --- | --- | --- |
| `get_env_config()` | `dict` — 地图 + 奖励 + 通信关键量 | `env.MultiRobotEnv` |
| `get_rl_config(**overrides)` | `dict` — 含 `_RL_DEFAULTS` 全集 | `main.py` / trainer |
| `get_channel_config()` | `_Args` 对象（兼容 `parse_args()`） | `communication/*` |
| `get_base_map_and_seed()` | `dict` — 生成器用 | `config.generator.main` |

`_Args` 是一个小型只读包装，让旧的 `parser.parse_args().xxx` 用法仍然成立；同时 `_parser_cache` / `_env_parser_cache` 是模块级缓存，避免重复读 YAML。

---

## 9. 配置一致性要点

- **`antenna_position` 优先级**：`channel.yml → map.yml → scenario.npz`。三处值不一致时，以 `channel.yml` 为准（若显式填写）。
- **`num_power_levels`**：训练纯导航阶段设为 `1`（动作空间 = 方向数）；引入功率选择时再调回 `3`。
- **`square_size_range`**：`map.yml` 为 `[3, 5]`；`yml_config.get_base_map_and_seed` 的默认回退值 `[7, 12]`。由于 `map.yml` 存在，此默认值不会生效，但修改 YAML 时请直接在 `map.yml` 中调整。
- **`P_min_diff`** 当前未被 `yml_config` 读取，若要在 SIC 功率分配中使用需显式扩展 `get_channel_config()`。

---

## 10. 典型调用流

```python
from config.yml_config import get_env_config, get_rl_config, get_channel_config

env_cfg = get_env_config()      # 首次调用若 scenario.npz 失效会自动重建
rl_cfg  = get_rl_config(lr=5e-5, num_episodes=200)  # CLI 覆盖
ch_cfg  = get_channel_config()  # noise_power_mw 已算好
```

手动重生成场景：
```bash
python -m config.generator.main
```
