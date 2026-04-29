# OFDMA 通信模型 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 OFDMA 通信模型，训练/测试时通过 `--comm_model {noma,ofdma}` 选择，默认 noma，对现有功能零侵入。

**Architecture:** 新建 `src/communication/ofdma.py` 实现与 NOMA 相同签名的 BER 奖励函数；`env.py` 初始化时按 `config["comm_model"]` 绑定 `self._compute_comm`；`main.py` 增加 CLI 参数写入 env_cfg。

**Tech Stack:** Python 3, NumPy, SciPy（已有依赖），pytest（tests/ 已有套件）

---

## 文件清单

| 操作 | 路径 | 职责 |
|---|---|---|
| 新建 | `src/communication/ofdma.py` | OFDMA BER/SNR 计算，无分簇/预编码/SIC |
| 修改 | `config/base/channel.yml` | 新增 `comm_model` 默认值 `noma` |
| 修改 | `src/config/yml_config.py` | `get_env_config()` 暴露 `comm_model` 字段 |
| 修改 | `src/env/env.py` | `__init__` 路由；`step()` 调 `self._compute_comm` |
| 修改 | `main.py` | 增加 `--comm_model` 参数，注入 env_cfg |
| 新建 | `tests/test_ofdma.py` | OFDMA 单元测试 |

---

## Task 1: 新建 `src/communication/ofdma.py`

**Files:**
- Create: `src/communication/ofdma.py`

- [ ] **Step 1: 写入 ofdma.py**

```python
"""
OFDMA 通信模型：每个 agent 占用独立正交子载波，无簇间/簇内干扰。
对齐原论文式 (2-17)(2-18)。
"""
import numpy as np
from communication.precompute import PrecomputedRadioMap
from communication.SIC import compute_ber, ber_to_reward


def compute_ber_rewards_ofdma(
    radio_map: PrecomputedRadioMap,
    positions,
    power_actions,
    P_sum,
    num_power_levels,
    N,
    D,
    noise_power,
    rng=None,
    ber_reward_min=-1.0,
    ber_reward_max=1.0,
    ber_worst=0.5,
    ber_best=1e-10,
):
    """
    OFDMA BER 奖励计算。

    Args:
        radio_map: PrecomputedRadioMap 实例
        positions: (K, 2) int array，agent 网格坐标
        power_actions: (K,) int array，功率等级索引
        P_sum: 总发射功率 (mW)
        num_power_levels: 可用功率等级数 p
        N: 信道块长度
        D: 数据包大小 (bits)
        noise_power: 总噪声功率 (mW)
        rng: numpy random generator

    Returns:
        dict with:
            ber:   (K,) 每个 agent 的 BER
            sinr:  (K,) 每个 agent 的 SNR（OFDMA 无干扰，即 SNR）
            reward:(K,) 每个 agent 的通信奖励
    """
    positions = np.array(positions, dtype=int)
    K = len(positions)

    # 每个 agent 分配的最大功率 = P_sum / K
    P_per_agent = P_sum / K

    # 可用功率等级（与 NOMA strong_powers 结构对称）
    p = num_power_levels
    power_levels = np.array([P_per_agent / (2 ** i) for i in range(1, p + 1)])  # 降序

    # 子载波噪声功率：总噪声 / K
    noise_per_sub = noise_power / K

    # 获取信道向量（含 Rayleigh 衰落）
    H = radio_map.get_channel_vectors(positions, rng=rng)  # (K, N_t)

    # 每个 agent 独立 SNR，无干扰
    # 等效信道增益 = |h_k|^2（直接用向量模平方，无需预编码）
    gains = np.sum(np.abs(H) ** 2, axis=1)  # (K,)

    # 按功率动作选取功率
    p_indices = np.minimum(power_actions, len(power_levels) - 1)
    powers = power_levels[p_indices]  # (K,)

    snr = powers * gains / noise_per_sub  # (K,)

    # 有限块长 BER：带宽缩小 K 倍，等效码率 = K*D/N
    effective_D = K * D
    ber_all = compute_ber(snr, N, effective_D)

    reward_all = ber_to_reward(
        ber_all,
        reward_min=ber_reward_min,
        reward_max=ber_reward_max,
        ber_worst=ber_worst,
        ber_best=ber_best,
    )

    return {
        "ber": ber_all,
        "sinr": snr,
        "reward": reward_all,
    }
```

- [ ] **Step 2: 验证文件可导入**

```bash
cd D:/MyProject/Python_Project/Graduation_Project
python -c "import sys; sys.path.insert(0,'src'); from communication.ofdma import compute_ber_rewards_ofdma; print('import OK')"
```

期望输出：`import OK`

- [ ] **Step 3: 提交**

```bash
git add src/communication/ofdma.py
git commit -m "feat(comm): add OFDMA BER reward module"
```

---

## Task 2: 新建 `tests/test_ofdma.py`

**Files:**
- Create: `tests/test_ofdma.py`

- [ ] **Step 1: 写入测试文件**

```python
"""单元测试：OFDMA 通信模型 (src/communication/ofdma.py)。"""
import numpy as np
import pytest


def _make_radio_map(K=4):
    """构造最小化 PrecomputedRadioMap（2x2 地图，不触碰缓存文件）。"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from communication.precompute import PrecomputedRadioMap
    return PrecomputedRadioMap(
        map_size=(10, 10),
        grid_size=0.4,
        antenna_position=(5, 5),
        h_AP=2.0,
        h_robot=1.5,
        h_block=3.0,
        n_antenna=8,
        carrier_freq_ghz=3.5,
        forbidden_areas=[],
        sigma_rayleigh=1.2,
        cache_dir=None,
    )


def test_ofdma_output_shape():
    """返回 dict 中 ber/sinr/reward 均为 (K,) 形状。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    K = 4
    positions = np.array([[1, 1], [2, 3], [5, 6], [7, 8]])
    power_actions = np.array([0, 1, 0, 1])
    result = compute_ber_rewards_ofdma(
        rm, positions, power_actions,
        P_sum=100.0, num_power_levels=2,
        N=256, D=16, noise_power=1e-7,
        rng=np.random.default_rng(42),
    )
    assert result["ber"].shape == (K,)
    assert result["sinr"].shape == (K,)
    assert result["reward"].shape == (K,)


def test_ofdma_ber_in_valid_range():
    """每个 agent 的 BER ∈ [0, 1]，不含 NaN/Inf。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    positions = np.array([[1, 2], [3, 4], [6, 7], [8, 9]])
    power_actions = np.zeros(4, dtype=int)
    result = compute_ber_rewards_ofdma(
        rm, positions, power_actions,
        P_sum=100.0, num_power_levels=1,
        N=256, D=16, noise_power=1e-7,
        rng=np.random.default_rng(0),
    )
    assert np.all(np.isfinite(result["ber"]))
    assert np.all(result["ber"] >= 0.0)
    assert np.all(result["ber"] <= 1.0)


def test_ofdma_snr_positive():
    """SNR > 0（功率和信道增益均为正）。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    positions = np.array([[2, 2], [4, 4]])
    power_actions = np.zeros(2, dtype=int)
    result = compute_ber_rewards_ofdma(
        rm, positions, power_actions,
        P_sum=100.0, num_power_levels=1,
        N=256, D=16, noise_power=1e-7,
        rng=np.random.default_rng(1),
    )
    assert np.all(result["sinr"] > 0)


def test_ofdma_higher_power_lower_ber():
    """更高功率等级 → 更高 SNR → 更低 BER（统计平均，固定种子）。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    positions = np.array([[3, 3], [6, 6]])
    rng = np.random.default_rng(99)
    result_high = compute_ber_rewards_ofdma(
        rm, positions, np.zeros(2, dtype=int),  # 最高功率（index=0）
        P_sum=200.0, num_power_levels=3,
        N=256, D=16, noise_power=1e-7, rng=rng,
    )
    rng2 = np.random.default_rng(99)
    result_low = compute_ber_rewards_ofdma(
        rm, positions, np.full(2, 2, dtype=int),  # 最低功率（index=2）
        P_sum=200.0, num_power_levels=3,
        N=256, D=16, noise_power=1e-7, rng=rng2,
    )
    # 平均 BER：高功率 ≤ 低功率（数值上留少量容差）
    assert result_high["ber"].mean() <= result_low["ber"].mean() + 1e-6


def test_ofdma_k1_works():
    """K=1 时不应报错（单 agent 边界条件）。"""
    from communication.ofdma import compute_ber_rewards_ofdma
    rm = _make_radio_map()
    positions = np.array([[4, 4]])
    result = compute_ber_rewards_ofdma(
        rm, positions, np.array([0]),
        P_sum=100.0, num_power_levels=1,
        N=256, D=16, noise_power=1e-7,
        rng=np.random.default_rng(7),
    )
    assert result["ber"].shape == (1,)
```

- [ ] **Step 2: 运行测试，确认全部通过**

```bash
cd D:/MyProject/Python_Project/Graduation_Project
python -m pytest tests/test_ofdma.py -v
```

期望：5 个测试全部 PASS。

- [ ] **Step 3: 提交**

```bash
git add tests/test_ofdma.py
git commit -m "test(comm): add OFDMA unit tests"
```

---

## Task 3: 在 `channel.yml` 和 `yml_config.py` 中暴露 `comm_model`

**Files:**
- Modify: `config/base/channel.yml`
- Modify: `src/config/yml_config.py:163-215`

- [ ] **Step 1: 在 `channel.yml` 末尾追加**

在 `config/base/channel.yml` 末尾（`num_power_levels` 之后）添加：

```yaml
comm_model:
  value: noma
  description: "通信多址方案: noma | ofdma"
```

- [ ] **Step 2: 在 `yml_config.py:get_env_config()` 中读取并暴露**

在 `src/config/yml_config.py` 的 `get_env_config()` 函数内，`return` 语句的 dict 中增加一行（紧接 `"noise_power_mw"` 行之后）：

```python
        "comm_model": str(_get_yml_value(ch, "comm_model", "noma")),
```

其中 `ch` 已在函数开头通过 `ch = _load_base_yml("channel")` 加载。完整 return dict 的末尾部分应如下：

```python
        "noise_power_mw": channel_cfg.noise_power_mw,
        "comm_model": str(_get_yml_value(_load_base_yml("channel"), "comm_model", "noma")),
        "random_seed": base["random_seed"],
    }
```

- [ ] **Step 3: 验证**

```bash
cd D:/MyProject/Python_Project/Graduation_Project
python -c "
import sys; sys.path.insert(0,'src')
from config.yml_config import get_env_config
cfg = get_env_config()
print('comm_model =', cfg['comm_model'])
assert cfg['comm_model'] in ('noma', 'ofdma')
print('OK')
"
```

期望输出：`comm_model = noma` 和 `OK`。

- [ ] **Step 4: 提交**

```bash
git add config/base/channel.yml src/config/yml_config.py
git commit -m "feat(config): expose comm_model in channel.yml and get_env_config()"
```

---

## Task 4: 修改 `src/env/env.py` 路由通信模型

**Files:**
- Modify: `src/env/env.py:21` (顶层 import)
- Modify: `src/env/env.py:80-103` (`__init__` 通信参数区域)
- Modify: `src/env/env.py:290` (`step()` 中调用处)

- [ ] **Step 1: 替换顶层 import**

将 `src/env/env.py` 第 21 行：

```python
from communication.ber_reward import compute_ber_rewards
```

删除（不需要了，改为 `__init__` 内动态绑定）。

- [ ] **Step 2: 在 `__init__` 中添加路由逻辑**

在 `src/env/env.py` 的 `__init__` 中，`self.radio_map = PrecomputedRadioMap(...)` 代码块结束后（约第 103 行），添加：

```python
        # 通信模型路由：noma（默认）或 ofdma
        _comm_model = str(config.get("comm_model", "noma")).lower()
        if _comm_model == "ofdma":
            from communication.ofdma import compute_ber_rewards_ofdma
            self._compute_comm = compute_ber_rewards_ofdma
        else:
            from communication.ber_reward import compute_ber_rewards
            self._compute_comm = compute_ber_rewards
        self.comm_model = _comm_model
```

同时在 `__init__` 末尾的 print 块中补充一行（`print("-" * 33)` 之前）：

```python
        print(f"  comm_model   : {self.comm_model}")
```

- [ ] **Step 3: 替换 `step()` 中的调用**

将 `src/env/env.py` 中 `step()` 的（约第 290 行）：

```python
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
```

替换为：

```python
            ber_result = self._compute_comm(
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
```

- [ ] **Step 4: 验证 NOMA 路径不变（运行现有 env 测试）**

```bash
cd D:/MyProject/Python_Project/Graduation_Project
python -m pytest tests/test_env.py -v
```

期望：原有测试全部 PASS（回归验证）。

- [ ] **Step 5: 验证 OFDMA 路径可用**

```bash
cd D:/MyProject/Python_Project/Graduation_Project
python -c "
import sys; sys.path.insert(0,'src')
from config.yml_config import get_env_config
from env.env import MultiRobotEnv
cfg = get_env_config()
cfg['comm_model'] = 'ofdma'
env = MultiRobotEnv(cfg)
states = env.reset()
actions = [0] * env.num_agents
ns, rs, ds, info = env.step(actions)
print('OFDMA step OK, ber =', info['ber'])
"
```

期望：打印出 `comm_model   : ofdma` 和 BER 值，不报错。

- [ ] **Step 6: 提交**

```bash
git add src/env/env.py
git commit -m "feat(env): route comm model via self._compute_comm (noma/ofdma)"
```

---

## Task 5: 在 `main.py` 中增加 `--comm_model` CLI 参数

**Files:**
- Modify: `main.py:44-90` (`_parse_args`)
- Modify: `main.py:263-283` (`main()` 中 env_cfg 构建)

- [ ] **Step 1: 在 `_parse_args()` 中添加参数**

在 `main.py` 的 `_parse_args()` 函数内，`--randomize_reset` 参数之后添加：

```python
    p.add_argument("--comm_model", choices=["noma", "ofdma"], default=None,
                   help="communication access scheme: noma (default) | ofdma")
```

- [ ] **Step 2: 在 `main()` 中将 CLI 值注入 env_cfg**

在 `main.py` 的 `main()` 函数中，`env = MultiRobotEnv(env_cfg, ...)` 之前，找到 `env_cfg = get_env_config()` 附近，添加：

```python
    # CLI 覆盖 comm_model（不传则保持 channel.yml 默认值）
    if args.comm_model is not None:
        env_cfg["comm_model"] = args.comm_model
```

- [ ] **Step 3: 冒烟验证**

```bash
cd D:/MyProject/Python_Project/Graduation_Project
python main.py --model dqn --mode train --comm_model ofdma --num_episodes 1 --num_iterations 1 --no_save_model --no_plot --no_test_after_train
```

期望：启动并完成 1 个 episode，日志中出现 `comm_model   : ofdma`，无报错。

再验证 NOMA 默认：

```bash
python main.py --model dqn --mode train --num_episodes 1 --num_iterations 1 --no_save_model --no_plot --no_test_after_train
```

期望：日志中出现 `comm_model   : noma`，行为与原版一致。

- [ ] **Step 4: 提交**

```bash
git add main.py
git commit -m "feat(cli): add --comm_model {noma,ofdma} argument"
```

---

## Task 6: 运行完整测试套件（回归验证）

**Files:** 无新建/修改

- [ ] **Step 1: 运行全部单元测试**

```bash
cd D:/MyProject/Python_Project/Graduation_Project
python -m pytest tests/test_communication.py tests/test_ofdma.py tests/test_env.py tests/test_config.py -v
```

期望：所有测试 PASS，无 FAIL/ERROR。

- [ ] **Step 2: 若有失败，逐条修复后重跑，直至全部绿**

- [ ] **Step 3: 最终提交**

```bash
git add -A
git status  # 确认只有预期中的文件
git commit -m "chore: verify all tests pass after OFDMA integration"
```

---

## 验收标准

- `python main.py --comm_model ofdma ...` 能正常训练/测试，日志显示 `comm_model: ofdma`
- `python main.py`（不加参数）行为与原版完全一致（noma 默认）
- `tests/test_ofdma.py` 全部 PASS
- `tests/test_env.py` 全部 PASS（回归）
- OFDMA 的平均 BER 高于 NOMA（符合论文图 4-14 结论，可通过短训练跑肉眼验证）
