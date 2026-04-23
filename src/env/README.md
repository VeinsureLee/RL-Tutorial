# `env/` 模块说明

多机器人通感协同导航环境。网格 120×60（物理 48 m × 24 m，单格 0.4 m），
每个 agent 的动作是 **(移动方向, 发射功率等级)** 的复合动作。
环境在每步同时：

1. 更新位置（边界 / 禁区 / 到达 / 普通移动四种情形）
2. 结合当前簇结构调用 `communication.compute_ber_rewards` 算 BER
3. 把导航 + 通信合成三段式奖励（`step / approach / comm`），分别暴露到 `info`

```
actions (K,) ──► decode (dir_idx, power_idx)
                │
                ▼
           位置更新（边界 / 禁区 / 到达 / 普通）
                │
                ▼
        approach_rewards + step_rewards
                │
                ▼
  PrecomputedRadioMap + compute_ber_rewards
                │
                ▼
    Δε → {better, worse, same}  ──►  comm_rewards = ω · Δ
                │
                ▼
      rewards = step + approach + comm
```

---

## 1. MDP 形式化

| 元素 | 表示 | 说明 |
| --- | --- | --- |
| 状态 $s_i^t$ | `int ∈ [0, R·C)` | 第 $i$ 个 agent 在 $t$ 时刻的离散格点索引，`pos_to_index(r,c) = r·C + c` |
| 状态空间 | $|\mathcal{S}| = R \cdot C = 7200$ | 网格总格数 |
| 联合状态 | $(s_1^t,\dots,s_K^t)$ | 每 agent 独立观测自身 index，K = `num_agents` |
| 动作 $a_i^t$ | `int ∈ [0, n_dirs·n_powers)` | 复合动作，`decode_action(a) = (a // n_powers, a % n_powers)` |
| 动作空间 | $|\mathcal{A}| = n_{\mathrm{dir}} \cdot n_{\mathrm{pow}}$ | 默认 5 × 1 = 5（纯导航阶段），或 5 × 3 = 15 |
| 奖励 | $r_i^t = r^{\mathrm{step}} + r^{\mathrm{app}} + r^{\mathrm{comm}}$ | 三路分解，见 §3 |
| 终止 | $s_i^t = \mathrm{target}_i$ | 单 agent 到达即 done；`all_done` 控制 episode 结束 |

**动作解码**：
$$
\mathrm{dir\_idx} = \lfloor a / n_{\mathrm{pow}} \rfloor,\qquad
\mathrm{pow\_idx} = a \bmod n_{\mathrm{pow}}
$$

方向集合来自 `env.yml:action_directions`，默认 5 个：
`{(0,1), (1,0), (0,-1), (-1,0), (0,0)}`，分别是 右 / 下 / 左 / 上 / 停。

---

## 2. 状态转移

对每个仍在线（`done_flags[i] = False`）的 agent：

$$
(r', c') = (r_t + \Delta r,\;c_t + \Delta c),\qquad (\Delta r, \Delta c) = \mathrm{directions}[\mathrm{dir\_idx}]
$$

分四种情形处理：

| 情形 | 判据 | 位置 | `approach_rewards` |
| --- | --- | --- | --- |
| **越界** | $r' \notin [0, R)$ 或 $c' \notin [0, C)$ | 不变 | `reward_forbidden` |
| **禁区** | $(r',c') \in \mathrm{forbidden\_set}$ | 回滚为 $(r_t, c_t)$ | `reward_forbidden` |
| **到达** | $(r',c') = \mathrm{target}_i$ | $(r', c')$，`done ← True` | `reward_goal` |
| **普通** | 其它 | $(r', c')$ | 见 §3.2 距离奖励 |

> **关键不变量**：禁区 / 越界分支**不做**通信奖励（`continue` 跳过 comm 块），`prev_ber` 不更新，下一步的增量以今天为新基线。

---

## 3. 奖励结构

每步总奖励三路叠加，每路 `info` 中单独返回：

$$
\boxed{\;r_i^t = r_i^{\mathrm{step}} + r_i^{\mathrm{app}} + r_i^{\mathrm{comm}}\;}
$$

### 3.1 时间惩罚 `step_rewards`

对所有仍在线的 agent：

$$
r_i^{\mathrm{step}} = \mathtt{reward\_step} \quad (\text{默认 } -1)
$$

### 3.2 导航奖励 `approach_rewards`

用**曼哈顿距离**判断远近：

$$
d_i^t = |r_i^t - r_i^{\star}| + |c_i^t - c_i^{\star}|,\qquad \Delta d = d_i^{t+1} - d_i^{t}
$$

$$
r_i^{\mathrm{app}} =
\begin{cases}
\mathtt{reward\_goal},   & (r',c') = (r^\star, c^\star) \\
\mathtt{reward\_forbidden}, & (r',c') \in \mathrm{forbidden\_set} \text{ 或越界} \\
\mathtt{reward\_closer}, & \Delta d < 0 \\
\mathtt{reward\_farther}, & \Delta d > 0 \\
\mathtt{reward\_same},   & \Delta d = 0
\end{cases}
$$

默认值：`goal=50, forbidden=-5, closer=+1, farther=-1, same=0`。

### 3.3 通信奖励 `comm_rewards` —— 增量 BER 奖励

环境保存上一步 BER 向量 `self.prev_ber`（首步为 `NaN`）。对当前步 BER 向量 $\varepsilon_t$：

$$
\mathrm{sign}\bigl(\varepsilon_t^i - \varepsilon_{t-1}^i\bigr)
\;\mapsto\;
\Delta_i =
\begin{cases}
\mathtt{ber\_reward\_better}, & \varepsilon_t^i < \varepsilon_{t-1}^i \\
\mathtt{ber\_reward\_worse}, & \varepsilon_t^i > \varepsilon_{t-1}^i \\
0, & \text{相等 / 首步 / done / 进禁区}
\end{cases}
$$

$$
\boxed{\;r_i^{\mathrm{comm}} = \omega \cdot \Delta_i\;}
$$

默认 `omega=1.0`，`ber_reward_better=+1`，`ber_reward_worse=-1`。
**注意**：`communication.compute_ber_rewards` 的 `reward` 字段（绝对奖励）在本环境中**被忽略**，
env 只取其 `ber` 字段做差分。

### 3.4 三路之和

```python
rewards[i] = step_rewards[i] + approach_rewards[i] + comm_rewards[i]
```

禁区 / 越界分支在算完 `step + forbidden` 后 `continue`，因此这些 agent 的 `comm_rewards[i] = 0`。

---

## 4. 通信子系统接入

### 4.1 预计算一次

`__init__` 中构造 `PrecomputedRadioMap`，把与位置相关的静态量全算好缓存。
传入：`(map_size, grid_size, antenna_position, h_AP, h_robot, h_block, n_antenna, carrier_freq_ghz, forbidden_areas, sigma_rayleigh)`。
具体公式见 `communication/README.md` §3。

### 4.2 每步调用

```python
ber_result = compute_ber_rewards(
    radio_map, positions[active], power_actions[active],
    P_sum, num_power_levels, N, D, noise_power, rng=self.rng,
)
```

只把**仍在线**的 agent 传进去，`done` 的 agent 不参与分簇（否则会浪费一个 NOMA 槽位）。
返回 `{ber, sinr, reward}`；env 只取 `ber` 与 `sinr`，填到 `info` 供训练曲线绘制。

### 4.3 随机源

`self.rng = np.random.default_rng(random_seed)`。Rayleigh 衰落的随机性全部由此 RNG 提供，
保证同一种子下整段 episode 的信道样本可复现。

---

## 5. 关键接口

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `reset()` | `list[int]` 长度 K | 重置位置 / done / 轨迹 / prev_ber；返回各 agent 初始 state index |
| `step(actions)` | `(next_states, rewards, dones, info)` | actions 为长度 K 的复合动作列表 |
| `pos_to_index(r, c)` | `int` | $r \cdot C + c$ |
| `index_to_pos(idx)` | `(r, c)` | 逆映射 |
| `decode_action(a)` | `(dir_idx, pow_idx)` | §1 公式 |
| `all_done` 属性 | `bool` | 所有 agent 都到达后为真 |

**`info` 字段**（每个均为 shape `(K,)`）：

| 键 | 含义 |
| --- | --- |
| `ber` | 各 agent 本步 BER；done 位置维持 0 |
| `sinr` | SINR 线性值 |
| `power_indices` | 本步选择的功率等级索引 |
| `step_rewards` | 时间惩罚 |
| `approach_rewards` | 导航奖励 |
| `comm_rewards` | 通信奖励（含 $\omega$） |

---

## 6. 禁区集合构造 `_build_forbidden_set`

支持三种输入格式（历史兼容）：

| 输入形状 | 解释 |
| --- | --- |
| `(r, c, w, h)` | 矩形，左上角 `(r,c)`、宽 `w`、高 `h` |
| `((r1, c1), (r2, c2))` | 两对角点的矩形 |
| `((r, c), size)` | 正方形，左上角 `(r,c)`、边长 `size`（**当前场景生成器的输出**） |

统一展开为格点集合 `self.forbidden_set: set[(int, int)]`，进禁区判定为 $O(1)$ 查表。

---

## 7. 可视化接口

| 方法 | 说明 |
| --- | --- |
| `render_nav_frame(ax, ...)` | 白底 + 黑色方块表示禁区 + 每 agent 起点 (◯) / 终点 (★) / 当前位置 / 轨迹线 |
| `render_signal_frame(ax, ...)` | 信号质量热力图：`-PL` 作为色强，1%/99% 百分位裁剪；禁区额外扣 20 dB 便于对比 |
| `save_nav_gif(path, frames_data)` | 轨迹采样最多 100 帧写成 GIF |
| `save_signal_gif(path, frames_data)` | 同上，叠在信号图上 |

`frames_data` 是每帧的 `(positions, done_flags, trajectories)` 元组列表，由 `tester.py` 提供。

---

## 8. 与复现对象的关系

### 8.1 对齐的部分

- **状态编码**：单 agent 视角的格点索引（论文 §3.1）。
- **动作空间为方向 × 功率复合**：论文 §3.2 把方向动作与功率动作离散化并共用同一套 Q 网络。
- **奖励分解**：论文把目标接近、撞墙 / 禁区、通信质量作为独立奖励项。
- **通信奖励为 $R_{\mathrm{rate}}$** 映射（论文式 3-2）。

### 8.2 本项目的改动

1. **通信奖励改为增量（delta）** — 论文直接用 $R_{\mathrm{rate}} = -\log_{10}(\varepsilon)$ 或其归一化版本，每步独立给分。
   本项目改为与上一步 BER 比较：`better / same / worse` 三档定额奖励。动机是让奖励信号稀疏、幅度可控、与导航奖励量级匹配。
2. **引入 `reward_step` 时间惩罚** — 论文未显式设此项；本项目加 $-1$ 的固定时间惩罚，促使策略最短路径完成任务。
3. **越界 → forbidden 等价化** — 论文里越界通常直接终止或回滚；本项目统一按 `reward_forbidden` 处理，行为与撞禁区一致，简化状态机。
4. **done agent 的特殊处理** — 已 done 的 agent 从 NOMA 分簇、通信奖励、导航奖励中完全退出；下一步继续占 done 不动。论文版本的退出逻辑未显式描述。
5. **Rayleigh 随机源固定** — env 自有 `rng`，与 Python / NumPy 全局随机状态解耦，保证相同种子下信道样本严格可复现。
6. **可视化双视图** — 论文一般只画轨迹；本项目同时渲染"导航地图"与"信号质量地图"，方便观察 agent 是否被吸引到高信号区。

---

## 9. 后续扩展大纲（仅列方向）

**I. 状态表征**
- 单 index → 带目标相对位置 / 邻居 / BER 历史的结构化观测
- CNN 栅格观测 vs 全连接 flat 观测
- 通信观测：是否显式暴露 $\beta, \theta, \mathrm{LOS}$ 给策略

**II. 动作空间**
- 方向 × 功率从笛卡尔积改为条件动作（先选方向再选功率，层级策略）
- 连续功率（actor-critic）与离散导航并存
- 加入「通信仅选功率、不移动」的 STAY 特化动作

**III. 奖励塑形**
- 绝对 BER 奖励 vs 增量 BER vs 阈值触发
- 通信奖励权重 $\omega$ 的扫描 / 自适应调度
- 目标稀疏奖励（仅到达给分）与密集奖励的折中

**IV. 终止与复位**
- 步数上限后软终止（按剩余距离扣分）
- 随机起点 / 多任务课程
- 异步 episode（agent 到达后立即复用）

**V. 观测 / 动力学扩展**
- 时间片长度（1 step = ? ms）与动力学
- agent 间碰撞模型（当前允许同格）
- 动态禁区 / 移动障碍

**VI. 场景规模**
- $K$ 规模的扫描（1, 2, 4, 8, 16 ...）
- 地图尺寸 / 禁区数量的敏感性
- 多 AP / Cell-Free 扩展

---

## 10. 已知遗留

- **曼哈顿 vs 欧氏距离**：`approach_rewards` 使用曼哈顿距离（与论文一致），但 `reward_goal` 判据是**精确到格**，斜向一格不算"接近"。
- **STAY 动作与 done**：STAY 不会触发"到达"判定，若 agent 起点恰在目标格，reset 后第一步无法 done；目前没有在 `reset` 中提前判 done。
- **`reward` 字段冗余**：`compute_ber_rewards` 返回的 `reward` 不被使用，保留作埋点。
- **`prev_ber` 的首步基线**：首步 `comm_rewards = 0`；如果 episode 很短，通信奖励的有效信号会被削掉一步，可能对 $K=1$ 短 episode 影响明显。
- **可视化依赖 Agg 后端**：模块顶层强制 `matplotlib.use("Agg")` 并关闭 `unicode_minus`，避免 Windows 字体警告；交互式显示需另行切换后端。
