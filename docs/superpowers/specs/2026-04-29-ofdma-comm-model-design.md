# OFDMA 通信模型设计文档

**日期**：2026-04-29  
**状态**：已批准  
**作者**：Claude Code + 用户

---

## 背景

当前代码仅实现了 NOMA（非正交多址）通信模型。原论文（参考文献：《工业互联网场景下基于强化学习的多机器人协同控制与通信资源分配算法研究》第 2.3.4 节）将 OFDMA 作为对比基线方案，需在训练/测试时可选择通信模型。

---

## 目标

- 新增 OFDMA 通信模型，与现有 NOMA 模型可在训练/测试时通过 CLI 参数切换。
- 对已有功能和实验结果零侵入（默认仍为 NOMA）。

---

## 数学模型（对齐论文式 2-17、2-18）

### OFDMA 信道与 SNR

系统共 K 个 agent，总带宽 B，总功率 P_sum。

每个 agent 分配带宽 B/K，对应子载波噪声功率：

```
noise_per_sub = noise_power / K
```

每个 agent 的每簇最大功率：

```
P_per_agent_max = P_sum / K
```

功率等级离散化（p 个等级，结构与 NOMA 对称）：

```
P_k ∈ {P_per_agent_max/2, P_per_agent_max/4, ..., P_per_agent_max/2^p}
```

agent k 的 SNR（无簇内/簇间干扰）：

```
SNR_k = P_k * |h_k|² / noise_per_sub        # 式 2-17
```

### OFDMA BER（有限块长）

带宽缩小 K 倍，等效码率升高：

```
rate = K * D / N
```

信道色散（同 NOMA）：

```
V = 1 - (1 + SNR_k)^(-2)
```

解码错误概率：

```
xi    = ln2 * sqrt(N / V) * (log2(1 + SNR_k) - rate)
BER_k = Q(xi) = 0.5 * erfc(xi / sqrt(2))              # 式 2-18
```

奖励映射复用现有 `ber_to_reward()`，无变化。

---

## 代码架构（方案 A：策略模式）

### 新增文件

- `src/communication/ofdma.py`：实现 `compute_ber_rewards_ofdma()`

### 修改文件

| 文件 | 改动 |
|---|---|
| `config/base/channel.yml` | 新增 `comm_model: {value: noma}` |
| `src/config/yml_config.py` | `get_env_config()` 暴露 `comm_model` 字段 |
| `src/env/env.py` | `__init__` 按 `comm_model` 绑定计算函数；`step()` 调用 `self._compute_comm` |
| `main.py` | 增加 `--comm_model {noma,ofdma}` 参数 |

### 接口约定

`compute_ber_rewards_ofdma()` 签名与 `compute_ber_rewards()` 完全相同：

```python
def compute_ber_rewards_ofdma(
    radio_map, positions, power_actions,
    P_sum, num_power_levels, N, D, noise_power,
    rng=None,
    ber_reward_min=-1.0, ber_reward_max=1.0,
    ber_worst=0.5, ber_best=1e-10,
) -> dict:  # {"ber": (K,), "sinr": (K,), "reward": (K,)}
```

### env.py 路由逻辑

```python
# __init__ 中
if config.get("comm_model", "noma") == "ofdma":
    from communication.ofdma import compute_ber_rewards_ofdma
    self._compute_comm = compute_ber_rewards_ofdma
else:
    from communication.ber_reward import compute_ber_rewards
    self._compute_comm = compute_ber_rewards

# step() 中（原 compute_ber_rewards 调用改为）
comm_result = self._compute_comm(
    self.radio_map, active_positions, power_actions,
    self.P_sum, self.n_powers, self.N, self.D,
    self.noise_power, rng=self.rng, ...
)
```

---

## CLI 用法

```bash
# NOMA（默认，向后兼容）
python main.py --model madqn --mode train

# OFDMA
python main.py --model madqn --mode train --comm_model ofdma
python main.py --model madqn --mode test  --comm_model ofdma --model_path <path>
```

---

## 不在本次范围内

- TDMA 或其他多址方式
- 动态子载波分配（当前为均等分配）
- 多天线 OFDMA（MIMO-OFDMA）

---

## 预期结果

- `--comm_model noma`（默认）：行为与改动前完全一致
- `--comm_model ofdma`：无 BD 预编码、无 SIC，每 agent 独立子载波，BER 通常高于 NOMA（与论文图 4-14 结论一致）
