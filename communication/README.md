# `communication/` 模块说明

通信子系统：实现从 agent 网格位置到每步 BER 奖励的完整物理层计算链路。
整体流水线（由 `env.env.MultiRobotEnv.step()` 在每个训练步调用一次）：

```
agent positions (K,2)
       │
       ▼
PrecomputedRadioMap.get_channel_vectors   ──►  H ∈ ℂ^{K×Nt}   (β · a(θ) · Rayleigh)
       │
       ▼
cluster_agents (按 ‖h_k‖² 降序配对)        ──►  M = ⌊K/2⌋ 个 (strong, weak) 对
       │
       ▼
matrix_cal (两次 SVD 的 BD 预编码)         ──►  w_m ∈ ℂ^{Nt×2}
       │
       ▼
compute_sinr (NOMA + SIC)                  ──►  SINR_strong, SINR_weak
       │
       ▼
compute_ber (有限块长 URLLC)               ──►  ε_k
       │
       ▼
ber_to_reward (−log₁₀ 映射 + 裁剪)         ──►  R_rate ∈ [reward_min, reward_max]
```

模块间依赖：`ber_reward.py` 是对外入口，它调用 `precompute`、`diagonalization_precoding`、`SIC`；
`channel.py` 提供独立于预计算表的路径损耗 / 信道向量函数，主要供测试与旧接口使用。

---

## 1. `utils.py` — 单位换算与通用工具

| 函数 | 公式 | 说明 |
| --- | --- | --- |
| `watt2dbm(W)` | $P_{\mathrm{dBm}} = 10 \log_{10}(1000\,W)$ | 瓦 → dBm |
| `dbm2watt(dbm)` | $W = 10^{\mathrm{dBm}/10}/1000$ | dBm → 瓦 |
| `distances_calculation(pts)` | $d_k = \lVert p_k - p_{\mathrm{AP}} \rVert_2$ | 2D 欧氏距离，AP 坐标取自 YAML |
| `judge_los_nlos(...)` | — | 占位 TODO，当前未使用（LOS/NLOS 由 `PrecomputedRadioMap` 计算） |

---

## 2. `channel.py` — 信道参数 / 向量 / 路径损耗（非缓存路径）

提供不依赖预计算缓存的"一次性"计算接口，以及按范数划分强弱用户组的 `channel_group`。

### 2.1 路径损耗 `path_loss / path_loss_batch / path_loss_from_states`

基于 3GPP-like 对数距离模型，载波频率 $f_c$（GHz），距离 $d$（m）：

$$
\text{PL}_{\text{LOS}}(d) = 31.87 + 21.50 \log_{10}(d) + 19.0 \log_{10}(f_c) \quad [\text{dB}]
$$

$$
\text{PL}_{\text{SL}}(d) = 33.00 + 25.50 \log_{10}(d) + 20.0 \log_{10}(f_c)
$$

$$
\text{PL}_{\text{NLOS}}(d) = \max\bigl(\text{PL}_{\text{LOS}}(d),\;\text{PL}_{\text{SL}}(d)\bigr)
$$

物理距离由离散网格坐标按 `grid_size`（默认 0.4 m）换算：

$$
d = \text{grid\_size}\cdot \max\!\Bigl(\sqrt{(x - a_x)^2 + (y - a_y)^2},\;10^{-6}\Bigr)
$$

### 2.2 信道参数 `channel_parameter`

把大尺度损耗与 Rayleigh 小尺度衰落合成到一个 dB 量：

$$
r \sim \text{Rayleigh}(\sigma),\quad
\chi_k = \text{PL}_k - 10 \log_{10}(r_k)\;[\text{dB}]
$$

下游用 $\beta_k = 10^{-\chi_k/20}$ 恢复线性幅度增益。

### 2.3 信道向量 `channel_vector`

ULA 天线阵列（间距半波长），到达角 $\theta_k$（默认 $\pi/2$ = broadside），
天线索引 $n \in \{0, 1, \dots, N_t-1\}$：

$$
\mathbf{a}(\theta_k) = \frac{1}{\sqrt{N_t}}\bigl[1,\;e^{-j\pi\sin\theta_k},\;\dots,\;e^{-j\pi(N_t-1)\sin\theta_k}\bigr]^{\!\top}
$$

$$
\mathbf{h}_k = \beta_k \cdot \mathbf{a}(\theta_k) \in \mathbb{C}^{N_t}
$$

### 2.4 到达角 `compute_arrival_angles`

ULA 沿 $x$ 轴排列，broadside 为 $y$ 轴正向：

$$
\theta_k = \operatorname{atan2}(\Delta x_k,\;\Delta y_k),\qquad
\Delta x_k = x_k - a_x,\;\Delta y_k = y_k - a_y
$$

### 2.5 信道分组 `channel_group`

按 $\|\mathbf{h}_k\|_2$ 降序排列后对半切分，前半数为"大组"（信道好 / 强用户），
后半数为"小组"（信道差 / 弱用户）。大组第 $k$ 个与小组第 $k$ 个配成同一簇。

> 注：`ber_reward.cluster_agents` 是等价但更直接的实现，使用 $\|\mathbf{h}_k\|^2$ 作为排序键，
> 当前流水线只走 `cluster_agents`，`channel_group` 主要保留给 `run_tests.py`。

---

## 3. `precompute.py` — 预计算无线电地图 `PrecomputedRadioMap`

**动机**：所有与位置相关的静态量（距离、AoA、LOS、路径损耗、阵列响应）只依赖地图，
不随 agent 动作变化。每步重算是浪费，故一次性计算 120×60 网格并缓存到
`config/dynamic/radio_map_cache.npz`，步进时只做查表 + Rayleigh。

### 3.1 几何量

网格中心物理坐标 $(g_x, g_y) = ((r+0.5)\cdot g_s,\;(c+0.5)\cdot g_s)$，AP 坐标
$(a_x, a_y, a_z)$，机器人高度 $h_r$：

$$
d_{r,c} = \sqrt{(g_x - a_x)^2 + (g_y - a_y)^2 + (h_r - a_z)^2},\quad
d_{r,c} \leftarrow \max(d_{r,c},\,0.1)
$$

$$
\theta_{r,c} = \operatorname{atan2}(g_y - a_y,\;g_x - a_x)
$$

### 3.2 LOS 判定 `_compute_los_grid`（AABB slab method）

障碍物框 $[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]$，从网格点到 AP 的参数射线
$\mathbf{p}(t) = (g_x,g_y) + t\cdot((a_x,a_y)-(g_x,g_y))$：

$$
t^{\text{in}} = \max\!\bigl(\min(t_{x1},t_{x2}),\,\min(t_{y1},t_{y2})\bigr),\quad
t^{\text{out}} = \min\!\bigl(\max(t_{x1},t_{x2}),\,\max(t_{y1},t_{y2})\bigr)
$$

射线与障碍物相交当且仅当 $t^{\text{in}} \le t^{\text{out}}$ 且 $t^{\text{out}} > 0$ 且 $t^{\text{in}} < 1$。
任一障碍物挡住即判 NLOS。

### 3.3 路径损耗与大尺度衰落

$$
\text{PL}_{r,c} =
\begin{cases}
\text{PL}_{\text{LOS}}(d_{r,c}), & \text{LOS} \\
\max\bigl(\text{PL}_{\text{LOS}},\,\text{PL}_{\text{SL}}\bigr), & \text{NLOS}
\end{cases},\qquad
\beta_{r,c} = 10^{-\text{PL}_{r,c}/20}
$$

> 注：这里 LOS 截距为 $31.84$（与 `channel.py` 的 $31.87$ 略有差异，保留作为两套
> 推导的历史遗留；数值影响极小）。

### 3.4 ULA 阵列响应

$$
\mathbf{a}_{r,c}[n] = \frac{1}{\sqrt{N_t}}\exp\!\bigl(-j\pi\,n\,\sin\theta_{r,c}\bigr),\quad n = 0,\dots,N_t-1
$$

所有静态量保存为 `distances / aoa / los_grid / path_loss / beta / steering_vectors`。

### 3.5 运行时信道向量 `get_channel_vectors`

查表得 $\beta_k, \mathbf{a}_k$，叠加复高斯（等价 Rayleigh 包络）小尺度衰落：

$$
\mathbf{g}_k = \frac{1}{\sqrt{2}}\bigl(\mathcal{N}(0,\sigma^2) + j\,\mathcal{N}(0,\sigma^2)\bigr)^{N_t}
$$

$$
\mathbf{h}_k = \beta_k \cdot \mathbf{a}_k \odot \mathbf{g}_k \in \mathbb{C}^{N_t}
$$

### 3.6 缓存失效机制

参数指纹 `_compute_param_hash` 对 `(map_size, grid_size, ap_grid, ap_pos_m, h_robot, h_block, n_antenna, carrier_freq_ghz, forbidden_areas)` 做 MD5。
任一参数变化 ⇒ 重算并覆盖 `.npz`。

---

## 4. `diagonalization_precoding.py` — Block Diagonalization 预编码

消除簇间干扰，令第 $m$ 簇的预编码向量 $\mathbf{w}_m$ 落在其它簇信道的零空间中。
对应论文式 (2-7) ~ (2-10)。

设共 $M$ 簇，第 $m$ 簇信道矩阵 $\mathbf{H}_m \in \mathbb{C}^{N_m\times N_t}$（本项目 $N_m = 2$）。

### 4.1 第一次 SVD：构造干扰零空间

堆叠其它簇的信道：

$$
\widetilde{\mathbf{H}}_m = \bigl[\mathbf{H}_1^{\!\top},\dots,\mathbf{H}_{m-1}^{\!\top},\mathbf{H}_{m+1}^{\!\top},\dots,\mathbf{H}_M^{\!\top}\bigr]^{\!\top}
$$

对其做 SVD：

$$
\widetilde{\mathbf{H}}_m = \widetilde{\mathbf{U}}_m\,\widetilde{\boldsymbol{\Sigma}}_m\,\bigl[\widetilde{\mathbf{V}}_m^{(1)},\;\widetilde{\mathbf{V}}_m^{(0)}\bigr]^{\mathrm{H}}
$$

其中 $\widetilde{\mathbf{V}}_m^{(0)}$ 是对应零奇异值的右奇异向量，构成零空间基——任何列向量
$\mathbf{v}\in\text{span}(\widetilde{\mathbf{V}}_m^{(0)})$ 都满足 $\widetilde{\mathbf{H}}_m\mathbf{v} = \mathbf{0}$。

### 4.2 第二次 SVD：在零空间内对本簇信道优化

令等效信道 $\mathbf{H}_m^{\text{eff}} = \mathbf{H}_m\widetilde{\mathbf{V}}_m^{(0)}$，再做 SVD 并取前 $N_m$ 个
右奇异向量 $\mathbf{V}_m^{(1)}$：

$$
\mathbf{w}_m = \widetilde{\mathbf{V}}_m^{(0)}\,\mathbf{V}_m^{(1)} \in \mathbb{C}^{N_t\times N_m}
$$

性质：对 $\forall i\neq m,\;\mathbf{H}_i\mathbf{w}_m \approx \mathbf{0}$，即簇间无干扰。

### 4.3 函数接口

| 函数 | 作用 |
| --- | --- |
| `matrix_cal(cluster_list, m)` | 返回第 $m$ 簇的 $\mathbf{w}_m \in \mathbb{C}^{N_t\times N_m}$；当 $\widetilde{\mathbf{H}}_m$ 行数为 0（单簇退化）时返回单位阵 |
| `build_W_matrix(cluster_list)` | 水平拼接所有 $\mathbf{w}_m$，返回 $\mathbf{W} = [\mathbf{w}_1,\dots,\mathbf{w}_M] \in \mathbb{C}^{N_t \times 2M}$ |

数值细节：有效秩判定 `rank = sum(S_t > 1e-10 * S_t[0])`，避免小奇异值引入数值零空间误差。

---

## 5. `SIC.py` — NOMA 功率等级、SINR、URLLC BER 与奖励

### 5.1 二元功率控制 `get_power_levels`

论文 3.2 节：每簇最大功率 $P_{\max}^{\text{cluster}}$，可选功率等级数 $p$。

$$
P_{\text{strong}}^{(i)} = \frac{P_{\max}^{\text{cluster}}}{2^i},\quad i = p, p+1, \dots, 2p-1
$$

$$
P_{\text{weak}}^{(i)} = \frac{P_{\max}^{\text{cluster}}}{2^i},\quad i = 1, 2, \dots, p
$$

信道好的用户分到更小功率（$1/2^p \sim 1/2^{2p}$ 段），信道差的用户分到更大功率
（$1/2 \sim 1/2^p$ 段），形成 NOMA 功率差并让 SIC 可行。

### 5.2 SINR `compute_sinr`（论文式 2-12, 2-13）

记等效信道增益 $g_k = |\mathbf{h}_k\mathbf{w}_m|^2$。强用户先解码并减去自身信号（成功的 SIC），
故其信干噪比无簇内干扰；弱用户受强用户信号干扰：

$$
\boxed{\;\text{SINR}_{\text{strong}} = \frac{P_{\text{strong}}\,g_{\text{strong}}}{\sigma_n^2}\;}
$$

$$
\boxed{\;\text{SINR}_{\text{weak}} = \frac{P_{\text{weak}}\,g_{\text{weak}}}{P_{\text{strong}}\,g_{\text{weak}} + \sigma_n^2}\;}
$$

### 5.3 有限块长 BER `compute_ber`（论文式 2-14 ~ 2-16，URLLC）

Shannon 容量 $C = \log_2(1 + \gamma)$，信道色散（channel dispersion）：

$$
V(\gamma) = 1 - (1+\gamma)^{-2}
$$

编码率 $R = D/N$（$D$ 为有效负载比特数，$N$ 为信道块长度）。有限块长下的解码错误
概率由 Polyanskiy–Poor–Verdú 正态近似给出：

$$
\xi = \ln(2)\,\sqrt{\frac{N}{V}}\,(C - R)
$$

$$
\boxed{\;\varepsilon \approx Q(\xi) = \tfrac{1}{2}\,\operatorname{erfc}\!\Bigl(\tfrac{\xi}{\sqrt{2}}\Bigr)\;}
$$

实现上截断到 $[10^{-20},\,1]$ 防数值下溢；$V$ 下限取 $10^{-10}$。

### 5.4 BER → 奖励 `ber_to_reward`

将 $\varepsilon$ 以 $-\log_{10}$ 映射后在 $[\varepsilon_{\text{worst}}, \varepsilon_{\text{best}}]$ 之间归一化，
再线性缩放到 $[R_{\min}, R_{\max}]$：

$$
r_{\text{raw}} = -\log_{10}(\varepsilon),\quad
r_{\text{w}} = -\log_{10}(\varepsilon_{\text{worst}}),\quad
r_{\text{b}} = -\log_{10}(\varepsilon_{\text{best}})
$$

$$
\tilde{r} = \operatorname{clip}\!\Bigl(\tfrac{r_{\text{raw}} - r_{\text{w}}}{r_{\text{b}} - r_{\text{w}}},\,0,\,1\Bigr)
$$

$$
\boxed{\;R_{\text{rate}} = R_{\min} + \tilde{r}\,(R_{\max} - R_{\min})\;}
$$

默认 $\varepsilon_{\text{worst}} = 0.5,\;\varepsilon_{\text{best}} = 10^{-10}$，$[R_{\min}, R_{\max}] = [-1, 1]$。
引入该归一化是因为原始 $-\log_{10}(\varepsilon)$ 的跨度过大（0.3–20），会掩盖导航奖励量级。

---

## 6. `ber_reward.py` — 通信奖励主入口

### 6.1 `cluster_agents(H, K)`

按 $\|\mathbf{h}_k\|^2$ 降序排列，$M = \lfloor K/2 \rfloor$ 对：第 $m$ 名与第 $M+m$ 名组成一簇：

$$
\text{strong}_m = \pi(m),\quad \text{weak}_m = \pi(M+m),\quad m = 0,\dots,M-1
$$

（$\pi$ 为降序排列索引。）$K$ 奇数时最后一个 agent 单独处理。

### 6.2 `compute_ber_rewards(...)` 完整流程

对每个 env step：

1. **查表 + 小尺度**：$\mathbf{H} = \text{radio\_map.get\_channel\_vectors(positions)}$。
2. **退化分支** $K = 1$：无分簇、无 BD，直接用 $g = \|\mathbf{h}\|^2$ 与 strong power 计算 $\gamma = P g/\sigma_n^2$，再走 BER / 奖励。
3. **分簇**：`cluster_agents` 得到 $M$ 对 (strong, weak)，每簇功率预算 $P_{\max}^{\text{cluster}} = P_{\text{sum}}/M$。
4. **BD 预编码**：对每簇叠放 $\mathbf{H}_m = [\mathbf{h}_{\text{strong}};\mathbf{h}_{\text{weak}}]$，调用 `matrix_cal` 得 $\mathbf{w}_m$。
5. **NOMA 广播**：取 $\mathbf{w}_m$ 第一列并归一化，两用户共享 $\mathbf{w}_m$；算 $g_{\text{strong}}, g_{\text{weak}}$。
6. **功率选择**：根据 agent 的离散功率动作索引 $p_s, p_w$，从功率表中查 $P_{\text{strong}}, P_{\text{weak}}$。
7. **SINR → BER → Reward**：按 §5 公式逐簇计算。奇数落单 agent 按单用户处理。
8. **返回** `{"ber": (K,), "sinr": (K,), "reward": (K,)}` 给 `env.step()`。

---

## 7. `run_tests.py` — 自测入口

| 函数 | 作用 |
| --- | --- |
| `test_channel` | 路径损耗 / 信道向量 / `channel_group` 烟雾测试 |
| `test_diag` | 验证 BD 后簇间残余干扰 $\|\mathbf{H}_i\mathbf{w}_m\|_F \approx 0,\,i\neq m$ |
| `test_sic` | 端到端 BD + NOMA + SIC，打印 strong / weak SINR（dB） |
| `test_ber_random_agents` | 按 `number_of_robots` 随机抽样网格点，跑 `get_ber_reward` 并打印统计 |

> ⚠️ 该文件当前导入了 `ber_reward.get_ber_reward` 与 `ber_reward._get_grid_size`，
> 但 `ber_reward.py` 现存接口是 `compute_ber_rewards`——`run_tests.py` 相对新代码已经过期，
> 直接 `python -m communication.run_tests` 会 ImportError。调试时改调 `compute_ber_rewards`。

---

## 8. 关键参数对照表（来自 `config/base/channel.yml` 与 `env.yml`）

| 符号 | 代码字段 | 含义 |
| --- | --- | --- |
| $N_t$ | `number_of_antenna` | AP 天线数（ULA） |
| $f_c$ | `carrier_frequency` | 载波频率（GHz） |
| $\sigma$ | `sigma_rayleigh` | Rayleigh 尺度参数 |
| $g_s$ | `grid_size` | 网格边长（m），120×60 × 0.4 m = 48 × 24 m |
| $P_{\text{sum}}$ | `P_sum` | 每步全系统总发射功率（mW） |
| $p$ | `num_power_levels` | 离散功率等级数 |
| $N$ | `N` | 信道块长度（符号数） |
| $D$ | `D` | 数据包大小（bits） |
| $\sigma_n^2$ | `noise_power` | 噪声功率（mW） |
| $(a_x,a_y)$ | `antenna_position` | AP 离散网格坐标 |
| $h_{\text{AP}}, h_r, h_b$ | `h_AP / h_robot / h_block` | AP/机器人/障碍物高度（m，仅用于 3D 距离与 LOS） |
| $\varepsilon_{\text{worst}}, \varepsilon_{\text{best}}, R_{\min}, R_{\max}$ | `ber_worst / ber_best / ber_reward_min / ber_reward_max` | BER→奖励 映射端点 |

---

## 9. 设计取舍与已知遗留

- **$K=1$ 特殊分支**：避免 `cluster_agents` 在 $K<2$ 时产生空簇；退化为单用户下行。
- **$K$ 奇数落单处理**：最后一名用户无配对，按单用户 SINR 处理（无 NOMA 增益）。
- **BD 列选择**：两用户共享 $\mathbf{w}_m$ 的第一列，而不是分别使用 $\mathbf{w}_m[:,0]$ 与 $\mathbf{w}_m[:,1]$。
  这是"NOMA-BD 混合"的简化：BD 消簇间干扰，簇内靠功率域 NOMA + SIC 区分用户。
- **`channel.py` 与 `precompute.py` 路径损耗截距差 (31.87 vs 31.84)**：保留两套常量，
  运行时只走 `precompute`，`channel` 仅供离线测试，数值差异 < 0.1 dB。
- **`run_tests.py` 接口不匹配**：需更新为 `compute_ber_rewards`，目前不可直接运行。
- **`judge_los_nlos` 在 `utils.py` 为占位**：实际 LOS 判定已迁移至 `PrecomputedRadioMap._compute_los_grid`。
