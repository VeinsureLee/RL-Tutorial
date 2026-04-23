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
ber_to_reward (−log₁₀ 归一化 + 线性缩放)   ──►  R_rate ∈ [R_min, R_max]
```

模块内依赖：`ber_reward.py` 是对外入口，它调用 `precompute`、`diagonalization_precoding`、`SIC`；
`channel.py` 提供独立于预计算表的路径损耗 / 信道向量函数，目前主要供离线测试与旧接口保留。

---

## 1. `utils.py` — 单位换算与通用工具

| 函数 | 公式 | 说明 |
| --- | --- | --- |
| `watt2dbm(W)` | $P_{\mathrm{dBm}} = 10 \log_{10}(1000\,W)$ | 瓦 → dBm |
| `dbm2watt(\mathrm{dBm})` | $W = 10^{\mathrm{dBm}/10}/1000$ | dBm → 瓦 |
| `distances_calculation(\mathrm{pts})` | $d_k = \lVert p_k - p_{\mathrm{AP}} \rVert_2$ | 2D 欧氏距离，AP 坐标取自 YAML |
| `judge_los_nlos(...)` | — | 占位 TODO，当前未使用（LOS/NLOS 已迁移至 `PrecomputedRadioMap`） |

---

## 2. `channel.py` — 路径损耗 / 信道向量 / 分组（非缓存路径）

提供不依赖预计算缓存的"一次性"计算接口，以及按范数划分强弱用户组的 `channel_group`。
运行时流水线**并不走这里**，仅作为对照实现与离线测试保留。

### 2.1 路径损耗 `path_loss / path_loss_batch / path_loss_from_states`

基于 3GPP-like 对数距离模型，载波频率 $f_c$（GHz），距离 $d$（m）：

$$
\mathrm{PL}_{\mathrm{LOS}}(d) = 31.87 + 21.50 \log_{10}(d) + 19.0 \log_{10}(f_c) \;\;[\mathrm{dB}]
$$

$$
\mathrm{PL}_{\mathrm{SL}}(d)  = 33.00 + 25.50 \log_{10}(d) + 20.0 \log_{10}(f_c)
$$

$$
\mathrm{PL}_{\mathrm{NLOS}}(d) = \max\bigl(\mathrm{PL}_{\mathrm{LOS}}(d),\;\mathrm{PL}_{\mathrm{SL}}(d)\bigr)
$$

物理距离由离散网格坐标按 `grid_size`（默认 0.4 m）换算：

$$
d = g_s \cdot \max\!\Bigl(\sqrt{(x - a_x)^2 + (y - a_y)^2},\;10^{-6}\Bigr)
$$

### 2.2 信道参数 `channel_parameter`

把大尺度损耗与 Rayleigh 小尺度衰落合成到一个 dB 量：

$$
r \sim \mathrm{Rayleigh}(\sigma),\quad
\chi_k = \mathrm{PL}_k - 10 \log_{10}(r_k)\;[\mathrm{dB}]
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

按 $\|\mathbf{h}_k\|_2$ 降序排列后对半切分，前半为"大组"（强用户），后半为"小组"（弱用户）。
大组第 $k$ 个与小组第 $k$ 个配成同一簇。

> 当前流水线使用 `ber_reward.cluster_agents`（等价但用 $\|\mathbf{h}_k\|^2$ 作排序键）；`channel_group` 保留给手工调试。

---

## 3. `precompute.py` — 预计算无线电地图 `PrecomputedRadioMap`

**动机**：所有与位置相关的静态量（距离、AoA、LOS、路径损耗、阵列响应）只依赖地图与天线，
不随 agent 动作变化。每步重算浪费；一次性计算 120×60 网格并缓存到
`config/dynamic/radio_map_cache.npz`，步进时只做查表 + Rayleigh。

### 3.1 几何量

网格中心物理坐标 $(g_x, g_y) = \bigl((r+0.5)\cdot g_s,\;(c+0.5)\cdot g_s\bigr)$，AP 坐标 $(a_x, a_y, a_z)$，机器人高度 $h_r$：

$$
d_{r,c} = \sqrt{(g_x - a_x)^2 + (g_y - a_y)^2 + (h_r - a_z)^2},\quad
d_{r,c} \leftarrow \max(d_{r,c},\,0.1)
$$

$$
\theta_{r,c} = \operatorname{atan2}(g_y - a_y,\;g_x - a_x)
$$

### 3.2 LOS 判定 `_compute_los_grid`（AABB slab method）

障碍物框 $[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]$，从网格点到 AP 的参数射线
$\mathbf{p}(t) = (g_x,g_y) + t\cdot\bigl((a_x,a_y)-(g_x,g_y)\bigr)$：

$$
t^{\mathrm{in}} = \max\!\bigl(\min(t_{x1},t_{x2}),\,\min(t_{y1},t_{y2})\bigr),\quad
t^{\mathrm{out}} = \min\!\bigl(\max(t_{x1},t_{x2}),\,\max(t_{y1},t_{y2})\bigr)
$$

射线与障碍物相交当且仅当 $t^{\mathrm{in}} \le t^{\mathrm{out}}$、$t^{\mathrm{out}} > 0$、$t^{\mathrm{in}} < 1$。任一障碍物挡住即判 NLOS。

### 3.3 路径损耗与大尺度衰落

$$
\mathrm{PL}_{r,c} =
\begin{cases}
\mathrm{PL}_{\mathrm{LOS}}(d_{r,c}), & \mathrm{LOS} \\
\max\bigl(\mathrm{PL}_{\mathrm{LOS}},\,\mathrm{PL}_{\mathrm{SL}}\bigr), & \mathrm{NLOS}
\end{cases},\qquad
\beta_{r,c} = 10^{-\mathrm{PL}_{r,c}/20}
$$

> LOS 截距此处取 $31.84$（与 `channel.py` 的 $31.87$ 有 0.03 dB 差异），保留作为两套推导的历史遗留。

### 3.4 ULA 阵列响应

$$
\mathbf{a}_{r,c}[n] = \frac{1}{\sqrt{N_t}}\exp\!\bigl(-j\pi\,n\,\sin\theta_{r,c}\bigr),\quad n = 0,\dots,N_t-1
$$

静态量保存为 `distances / aoa / los_grid / path_loss / beta / steering_vectors`。

### 3.5 运行时信道向量 `get_channel_vectors`

查表得 $\beta_k, \mathbf{a}_k$，叠加复高斯（等价 Rayleigh 包络）小尺度衰落：

$$
\mathbf{g}_k = \frac{1}{\sqrt{2}}\bigl(\mathcal{N}(0,\sigma^2) + j\,\mathcal{N}(0,\sigma^2)\bigr),\quad \mathbf{g}_k \in \mathbb{C}^{N_t}
$$

$$
\mathbf{h}_k = \beta_k \cdot \mathbf{a}_k \odot \mathbf{g}_k \in \mathbb{C}^{N_t}
$$

### 3.6 缓存失效机制

参数指纹 `_compute_param_hash` 对 `(map_size, grid_size, ap_grid, ap_pos_m, h_robot, h_block, n_antenna, carrier_freq_ghz, forbidden_areas)` 做 MD5。
任一参数变化 ⇒ 重算并覆盖 `.npz`。

---

## 4. `diagonalization_precoding.py` — Block Diagonalization 预编码

消除簇间干扰，令第 $m$ 簇的预编码矩阵 $\mathbf{w}_m$ 落在**其它簇信道的零空间**中。对应论文式 (2-7) ~ (2-10)。

设共 $M$ 簇，第 $m$ 簇信道矩阵 $\mathbf{H}_m \in \mathbb{C}^{N_m\times N_t}$（本项目 $N_m = 2$）。

### 4.1 第一次 SVD：构造干扰零空间

堆叠其它簇的信道：

$$
\widetilde{\mathbf{H}}_m = \bigl[\mathbf{H}_1^{\mathrm{H}},\dots,\mathbf{H}_{m-1}^{\mathrm{H}},\mathbf{H}_{m+1}^{\mathrm{H}},\dots,\mathbf{H}_M^{\mathrm{H}}\bigr]^{\mathrm{H}}
$$

对其做 SVD：

$$
\widetilde{\mathbf{H}}_m = \widetilde{\mathbf{U}}_m\,\widetilde{\boldsymbol{\Sigma}}_m\,\bigl[\widetilde{\mathbf{V}}_m^{(1)},\;\widetilde{\mathbf{V}}_m^{(0)}\bigr]^{\mathrm{H}}
$$

其中 $\widetilde{\mathbf{V}}_m^{(0)}$ 是对应零奇异值的右奇异向量，构成零空间基——任何
$\mathbf{v}\in\mathrm{span}(\widetilde{\mathbf{V}}_m^{(0)})$ 都满足 $\widetilde{\mathbf{H}}_m\mathbf{v} = \mathbf{0}$。

### 4.2 第二次 SVD：在零空间内对本簇信道优化

令等效信道 $\mathbf{H}_m^{\mathrm{eff}} = \mathbf{H}_m\widetilde{\mathbf{V}}_m^{(0)}$，再做 SVD 并取前 $N_m$ 个右奇异向量 $\mathbf{V}_m^{(1)}$：

$$
\mathbf{w}_m = \widetilde{\mathbf{V}}_m^{(0)}\,\mathbf{V}_m^{(1)} \in \mathbb{C}^{N_t\times N_m}
$$

性质：对 $\forall i\neq m$，$\mathbf{H}_i\mathbf{w}_m \approx \mathbf{0}$，即簇间无干扰。

### 4.3 函数接口

| 函数 | 作用 |
| --- | --- |
| `matrix_cal(cluster_list, m)` | 返回第 $m$ 簇的 $\mathbf{w}_m \in \mathbb{C}^{N_t\times N_m}$；$\widetilde{\mathbf{H}}_m$ 行数为 0 时（单簇退化）返回单位阵 |
| `build_W_matrix(cluster_list)` | 水平拼接所有 $\mathbf{w}_m$，返回 $\mathbf{W} = [\mathbf{w}_1,\dots,\mathbf{w}_M] \in \mathbb{C}^{N_t \times 2M}$ |

数值细节：有效秩判定 `rank = sum(S_t > 1e-10 * S_t[0])`，避免小奇异值引入伪零空间。

---

## 5. `SIC.py` — 功率等级、SINR、URLLC BER 与奖励

### 5.1 二元功率控制 `get_power_levels`

论文 3.2 节：每簇最大功率 $P_{\max}^{\mathrm{cluster}}$，可选功率等级数 $p$。

$$
P_{\mathrm{strong}}^{(i)} = \frac{P_{\max}^{\mathrm{cluster}}}{2^i},\quad i = p,\,p+1,\,\dots,\,2p-1
$$

$$
P_{\mathrm{weak}}^{(i)} = \frac{P_{\max}^{\mathrm{cluster}}}{2^i},\quad i = 1,\,2,\,\dots,\,p
$$

信道好的用户分到更小功率（$1/2^p \sim 1/2^{2p}$），信道差的用户分到更大功率（$1/2 \sim 1/2^p$），
形成 NOMA 功率差让 SIC 可行。

### 5.2 SINR `compute_sinr`（论文式 2-12, 2-13）

记等效信道增益 $g_k = |\mathbf{h}_k\mathbf{w}_m|^2$。强用户先解码并减去自身信号（SIC 成功），
故其信干噪比无簇内干扰；弱用户受强用户信号干扰：

$$
\boxed{\;\mathrm{SINR}_{\mathrm{strong}} = \frac{P_{\mathrm{strong}}\,g_{\mathrm{strong}}}{\sigma_n^2}\;}
$$

$$
\boxed{\;\mathrm{SINR}_{\mathrm{weak}} = \frac{P_{\mathrm{weak}}\,g_{\mathrm{weak}}}{P_{\mathrm{strong}}\,g_{\mathrm{weak}} + \sigma_n^2}\;}
$$

### 5.3 有限块长 BER `compute_ber`（论文式 2-14 ~ 2-16，URLLC）

Shannon 容量 $C = \log_2(1 + \gamma)$，信道色散（channel dispersion）：

$$
V(\gamma) = 1 - (1+\gamma)^{-2}
$$

编码率 $R = D/N$（$D$ 有效负载比特数，$N$ 信道块长度）。有限块长下的解码错误概率由
Polyanskiy–Poor–Verdú 正态近似给出：

$$
\xi = \ln(2)\,\sqrt{\frac{N}{V}}\,(C - R)
$$

$$
\boxed{\;\varepsilon \approx Q(\xi) = \tfrac{1}{2}\,\operatorname{erfc}\!\Bigl(\tfrac{\xi}{\sqrt{2}}\Bigr)\;}
$$

实现上截断到 $[10^{-20},\,1]$ 防数值下溢；$V$ 下限取 $10^{-10}$。

### 5.4 BER → 奖励 `ber_to_reward`

将 $\varepsilon$ 以 $-\log_{10}$ 映射后在 $[\varepsilon_{\mathrm{worst}}, \varepsilon_{\mathrm{best}}]$ 之间归一化，
再线性缩放到 $[R_{\min}, R_{\max}]$：

$$
r_{\mathrm{raw}} = -\log_{10}(\varepsilon),\quad
r_{\mathrm{w}} = -\log_{10}(\varepsilon_{\mathrm{worst}}),\quad
r_{\mathrm{b}} = -\log_{10}(\varepsilon_{\mathrm{best}})
$$

$$
\tilde{r} = \operatorname{clip}\!\Bigl(\tfrac{r_{\mathrm{raw}} - r_{\mathrm{w}}}{r_{\mathrm{b}} - r_{\mathrm{w}}},\;0,\;1\Bigr)
$$

$$
\boxed{\;R_{\mathrm{rate}} = R_{\min} + \tilde{r}\,(R_{\max} - R_{\min})\;}
$$

默认 $\varepsilon_{\mathrm{worst}} = 0.5,\;\varepsilon_{\mathrm{best}} = 10^{-10}$，$[R_{\min}, R_{\max}] = [-1, 1]$。
引入该归一化是因为原始 $-\log_{10}(\varepsilon)$ 的跨度过大（$0.3$–$20$），会掩盖导航奖励量级。

---

## 6. `ber_reward.py` — 通信奖励主入口

### 6.1 `cluster_agents(H, K)`

按 $\|\mathbf{h}_k\|^2$ 降序排列，$M = \lfloor K/2 \rfloor$ 对：第 $m$ 名与第 $M+m$ 名组成一簇：

$$
\mathrm{strong}_m = \pi(m),\quad \mathrm{weak}_m = \pi(M+m),\quad m = 0,\dots,M-1
$$

（$\pi$ 为降序排列索引。）$K$ 奇数时最后一个 agent 单独处理。

### 6.2 `compute_ber_rewards(...)` 完整流程

对每个 env step：

1. **查表 + 小尺度**：$\mathbf{H} = \mathrm{radio\_map.get\_channel\_vectors(positions)}$。
2. **退化分支** $K = 1$：无分簇、无 BD，直接用 $g = \|\mathbf{h}\|^2$、$\gamma = P g / \sigma_n^2$，再走 BER / 奖励。
3. **分簇**：`cluster_agents` 得到 $M$ 对 (strong, weak)，每簇功率预算 $P_{\max}^{\mathrm{cluster}} = P_{\mathrm{sum}}/M$。
4. **BD 预编码**：对每簇叠放 $\mathbf{H}_m = [\mathbf{h}_{\mathrm{strong}};\mathbf{h}_{\mathrm{weak}}]$，调用 `matrix_cal` 得 $\mathbf{w}_m$。
5. **NOMA 广播**：取 $\mathbf{w}_m$ 第一列并归一化，两用户共享该向量；算 $g_{\mathrm{strong}}, g_{\mathrm{weak}}$。
6. **功率选择**：根据 agent 的离散功率动作索引 $p_s, p_w$，从功率表中查 $P_{\mathrm{strong}}, P_{\mathrm{weak}}$。
7. **SINR → BER → Reward**：按 §5 公式逐簇计算。奇数落单 agent 按单用户处理。
8. **返回** `{"ber": (K,), "sinr": (K,), "reward": (K,)}` 给 `env.step()`。

---

## 7. 关键参数对照表（来自 `config/base/channel.yml` 与 `env.yml`）

| 符号 | 代码字段 | 含义 |
| --- | --- | --- |
| $N_t$ | `number_of_antenna` | AP 天线数（ULA） |
| $f_c$ | `carrier_frequency` | 载波频率（GHz） |
| $\sigma$ | `sigma_rayleigh` | Rayleigh 尺度参数 |
| $g_s$ | `grid_size` | 网格边长（m），120×60 × 0.4 m = 48 × 24 m |
| $P_{\mathrm{sum}}$ | `P_sum` | 全系统总发射功率（mW） |
| $p$ | `num_power_levels` | 离散功率等级数 |
| $N$ | `channel_block_length` | 信道块长度（符号数） |
| $D$ | `packet_size` | 数据包大小（bits） |
| $\sigma_n^2$ | `noise_power_mw` | 噪声功率（mW） |
| $(a_x,a_y)$ | `antenna_position` | AP 离散网格坐标 |
| $h_{\mathrm{AP}}, h_r, h_b$ | `h_AP / h_robot / h_block` | AP / 机器人 / 障碍物高度（m） |
| $\varepsilon_{\mathrm{worst}}, \varepsilon_{\mathrm{best}}, R_{\min}, R_{\max}$ | `ber_worst / ber_best / ber_reward_min / ber_reward_max` | BER → 奖励 映射端点 |

---

## 8. 与复现对象的关系与优化

复现来源：《工业互联网场景下基于强化学习的多机器人协同控制与通信资源分配算法研究》（2024）。
本模块在**系统建模**上严格对齐论文 §2，以下逐项列出**保持**的部分与**本项目的改动**。

### 8.1 完全对齐的部分

| 论文位置 | 本模块实现 | 对应公式 |
| --- | --- | --- |
| 式 (2-1)~(2-3) 大尺度路径损耗（LOS / SL / NLOS） | `channel.path_loss*` / `precompute._compute_all` | §2.1, §3.3 |
| §2.2 ULA 阵列响应与信道向量 | `channel.channel_vector`, `precompute.steering_vectors` | §2.3, §3.4 |
| §2.3.1 NOMA 信道增益降序配对 | `ber_reward.cluster_agents` | §6.1 |
| 式 (2-7)~(2-10) 双 SVD BD 预编码 | `diagonalization_precoding.matrix_cal` | §4 |
| 式 (2-12)(2-13) SIC 强 / 弱用户 SINR | `SIC.compute_sinr` | §5.2 |
| 式 (2-14)~(2-16) 有限块长解码错误概率 | `SIC.compute_ber` | §5.3 |
| §3.2 二元功率控制等级 | `SIC.get_power_levels` | §5.1 |

### 8.2 本项目的改动 / 优化

1. **预计算无线电地图（`precompute.py`）** — 论文按步实时算距离 / AoA / LOS / PL。本项目对 120×60 网格一次性算好缓存为 NPZ，在参数哈希命中时零成本复用。每 step 只剩查表 + Rayleigh 采样，显著降低训练耗时（典型 30×+）。
2. **LOS 判定升级** — 论文为定性描述；本项目实现 2D AABB slab 射线相交（§3.2），对任意矩形禁区都可计算。
3. **BER → 奖励映射** — 论文式 (3-2) 直接用 $R_{\mathrm{rate}} = -\log_{10}(\varepsilon)$，跨度 $0.3 \sim 20$。本项目改为两端点归一化 + 线性缩放至 $[R_{\min}, R_{\max}] = [-1, 1]$（§5.4），避免通信奖励量级掩盖导航奖励。
4. **分簇排序键微调** — 论文按 $|\mathbf{h}_k|$ 排序（等价于 $\|\mathbf{h}_k\|$，`channel_group` 仍保留）；主流水线 `cluster_agents` 改用 $\|\mathbf{h}_k\|^2$，避免一次 sqrt，数值等价。
5. **$K=1$ 与 $K$ 奇数的退化处理** — 论文默认 $K = 2M$ 且 $K \ge 2$。本项目在 `compute_ber_rewards` 中补了：
   - $K=1$：直接 $\gamma = P\,\|\mathbf{h}\|^2/\sigma_n^2$，走单用户 URLLC BER。
   - $K$ 奇数：末位 agent 单用户处理，不参与 NOMA。
6. **BER 奖励在环境层改为增量（delta）奖励** — 论文用绝对 BER → 奖励。`env.step()` 最终用的是 $\operatorname{sign}(\varepsilon_t - \varepsilon_{t-1})$ 映射到 `{better, same, worse}`，这部分逻辑不在 `communication/` 内、在 `env/env.py`，但会影响本模块的语义：这里输出的 `reward` 字段在当前 env 中**未被直接累加**，只有 `ber` 字段被取来算增量。
7. **NOMA + BD 混合的 `w_m` 列选取** — 论文 BD 给每簇两用户分别分配列向量；本项目简化为两用户共享 $\mathbf{w}_m[:,0]$ 归一化后的广播向量，簇内靠功率域 NOMA 区分用户。

---

## 9. 后续扩展大纲（仅列方向，不做详细讨论）

本节只点出未来可扩展与可对比实验的方向，具体取舍与公式放到各自专题文档中。

**I. 分簇算法**
- 两两配对 vs $K>2$ 成员大簇的 NOMA
- 信道增益排序 vs 基于相关性 / 角度分离度的聚类
- 分簇策略的必要性消融（与随机配对、与 OFDMA 独占对比）

**II. 预编码**
- BD vs MMSE、ZF、WMMSE 的对比
- 列选取（共享第一列 vs 独立列）与公平性
- 低天线数或相关信道下的 BD 数值退化

**III. 功率控制**
- 二元等级数 $p$ 的影响（$p=1,2,3$ 消融）
- 连续功率域（actor-critic / continuous action）对比
- 簇间功率再分配（非均匀 $P_{\max}^{\mathrm{cluster}}$）

**IV. BER 奖励塑形**
- 绝对奖励 vs 增量奖励 vs 阈值奖励
- 端点 $\varepsilon_{\mathrm{worst}}, \varepsilon_{\mathrm{best}}$ 与 $\omega$ 的耦合效应
- 按用户角色（强 / 弱）差异化加权

**V. 信道与物理层扩展**
- UPA / 稀疏天线阵列替代 ULA
- 多 AP / Cell-Free MIMO
- 非 Rayleigh 小尺度（Rician / Nakagami）
- 3D LOS 判定（考虑 `h_block`）

**VI. 与基线的系统级对比**
- NOMA vs OFDMA vs TDMA
- DQN / DDQN / Dueling / Rainbow 在同一信道下的差异
- 通信-导航奖励权重 $\omega$ 扫描

---

## 10. 设计取舍与已知遗留

- **$K=1$ 特殊分支**：避免 `cluster_agents` 在 $K<2$ 时产生空簇；退化为单用户下行。
- **$K$ 奇数落单处理**：最后一名 agent 无配对，按单用户 SINR 处理（无 NOMA 增益）。
- **BD 列选择**：两用户共享 $\mathbf{w}_m$ 第一列，而不是分别使用 $\mathbf{w}_m[:,0]$ 与 $\mathbf{w}_m[:,1]$。
  这是"NOMA-BD 混合"简化：BD 消簇间干扰，簇内靠功率域 NOMA + SIC 区分用户。
- **`channel.py` 与 `precompute.py` 路径损耗截距差（31.87 vs 31.84）**：保留两套常量。
  运行时只走 `precompute`，`channel` 仅供离线测试，数值差 $<0.1$ dB。
- **`judge_los_nlos` 在 `utils.py` 为占位**：实际 LOS 判定已迁移至 `PrecomputedRadioMap._compute_los_grid`。
- **`reward` 字段的语义差**：`compute_ber_rewards` 返回的 `reward` 对应绝对映射；env 层另起炉灶用 $\Delta\varepsilon$ 重新打分。
  接口未删，后续若切回绝对奖励可直接复用。
- **`P_min_diff`（channel.yml）未被读取**：`yml_config.get_channel_config()` 没有加载这一项，功率表目前按 $2^i$ 几何级数生成；需接入时补字段。
