# Unit Tests

## 运行

```bash
# 全量
pytest

# 生成 HTML 报告（按日期归档）
pytest --html=experiments/unit_tests/$(date +%Y%m%d)/pytest.html --self-contained-html

# 单模块
pytest tests/test_config.py -v
```

## 测试覆盖

| 文件 | 覆盖模块 | 关键测试点 |
|---|---|---|
| [`test_config.py`](test_config.py) | `src/config/` | YAML 加载；scenario.npz 重建触发；radio_map 哈希失效 |
| [`test_env.py`](test_env.py) | `src/env/` | step 四情形奖励；done agent 跳过；prev_ber 首步 NaN |
| [`test_communication.py`](test_communication.py) | `src/communication/` | BD 零空间验证；URLLC BER 单点对齐；K=1 退化路径 |
| [`test_rl_algorithms.py`](test_rl_algorithms.py) | `src/rl_algorithms/` | ReplayBuffer 采样；epsilon 衰减；目标网络同步 |

## 写新测试的规范

- 单元测试只覆盖**模块内部逻辑**，不做端到端训练
- `conftest.py` 提供最小 env / model fixture
- 避免依赖真实 `config/dynamic/scenario.npz`，用 `tmp_path` 构造临时场景
- 固定随机种子（`np.random.seed(0)` / `torch.manual_seed(0)`）

## 骨架状态

- [ ] `test_config.py`
- [ ] `test_env.py`
- [ ] `test_communication.py`
- [ ] `test_rl_algorithms.py`
- [ ] `conftest.py`
