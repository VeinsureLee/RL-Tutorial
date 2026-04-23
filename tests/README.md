# tests/

三类 pytest 用例：

| 类别 | 文件 | 用途 |
| --- | --- | --- |
| **正确性** | `test_config.py` / `test_env.py` / `test_communication.py` / `test_rl_algorithms.py` | 覆盖各模块关键不变量（step 分解、BD 零空间、RunContext…） |
| **性能** | `test_env_performance.py` | env.step / reset / radio_map / BER 计算的耗时，给出数字 + 断言上限 |
| **可视化** | `test_env_visualization.py` | 脚本化动作序列（上走 10、方形回路、撞禁区…），pytest 断言行为 + 保存 PNG 轨迹图肉眼验证 |

辅助：`_helpers.py`（自定义 env、脚本化、时间计量、轨迹图保存）、`conftest.py`（sys.path + 共享 fixture）。

---

## 最省事：点 ▶ `run_tests.py`

项目根的 [`run_tests.py`](../run_tests.py) 就是 VSCode 单按钮入口：
1. 打开 `run_tests.py`
2. 右上角 ▶ **Run Python File**
3. 看终端：每条 PASS/FAIL，性能会打印 `[PERF] ... ms`，可视化会打印 PNG 保存路径

完成后产物都在 `experiments/unit_tests/<today>/`：

```
experiments/unit_tests/20260423/
├── pytest.html      # 完整报告，浏览器直接打开
└── viz/             # 轨迹 PNG（VSCode 里点图即可预览）
    ├── up10.png
    ├── down10.png
    ├── L_shape.png
    ├── square_loop.png
    ├── hit_forbidden.png
    └── ...
```

---

## 聚焦某一类

编辑 `run_tests.py` 顶部的 `PATTERN`：

```python
PATTERN = "test_env_visualization.py"   # 只跑可视化
PATTERN = "test_env_performance.py"     # 只跑性能
PATTERN = "test_env*.py"                # env 正确性 + 性能 + 可视化
PATTERN = "test_*.py"                   # 全跑（默认）
```

或者不改文件，命令行：
```bash
.venv/Scripts/python -m pytest tests/test_env_visualization.py -v -s
```

---

## 性能测试解读

运行后终端里会看到：
```
[PERF] env.step (steady state)  mean=0.166 ms  median=0.163 ms  ...
[PERF] env.reset                 mean=0.007 ms  ...
[PERF] compute_ber_rewards (K=1) mean=0.062 ms  ...
[PERF] env.step throughput       6821 steps/sec  (500 steps in 73 ms)
```

断言的上限在 `test_env_performance.py` 顶部：
```python
ENV_STEP_MS_MAX = 20.0         # 单步 env.step 稳态
ENV_RESET_MS_MAX = 5.0
COMPUTE_BER_MS_MAX = 15.0
RADIO_MAP_CACHED_INIT_MS_MAX = 800.0
```

如果你在不同机器上跑觉得某条太紧 / 太松，直接改数字。

**退化 / regression 检测**：以后改了代码，跑一次 pytest 看耗时有没有显著变化。如果
`env.step` 原来 0.17ms，改完突然 2ms，马上能看出来是哪类新增开销。

---

## 可视化测试解读

每个 test 做三件事：

1. 把 env 的 `start_states` / `target_states` / `forbidden_set` 直接覆盖到测试所需的
   场景（**不重算** radio_map，所以很快）
2. 跑一段方向序列（比如 `[DIR_UP] * 10`）
3. 断言若干关键位置（起点、拐点、终点）+ 保存 PNG 到
   `experiments/unit_tests/<today>/viz/`

pytest 过 = 行为正确；PNG 是眼见为实。两者都通过才算这个场景真的对。

现有场景清单：

| 测试 | 场景 | 期望终点 |
| --- | --- | --- |
| `test_viz_up_10` | (30,30) → UP × 10 | (20, 30) |
| `test_viz_down_10` | (30,30) → DOWN × 10 | (40, 30) |
| `test_viz_right_15` | (60,10) → RIGHT × 15 | (60, 25) |
| `test_viz_left_15` | (60,45) → LEFT × 15 | (60, 30) |
| `test_viz_L_shape` | (30,20) → R×10 + D×10 | (40, 30)，拐点 (30,30) |
| `test_viz_square_loop` | R5 D5 L5 U5 | 回到起点 (30, 30) |
| `test_viz_hit_top_boundary` | (2,30) → UP × 5 | 停在 (0, 30) |
| `test_viz_hit_forbidden` | (20,18) → RIGHT × 5，禁区 (20..22, 20..22) | 停在 (20, 19) |
| `test_viz_reach_target` | 目标 (30,35)，RIGHT × 8 | 第 5 步 done |
| `test_viz_stay` | STAY × 5 | 原地不动 |

### 加新场景

在 `test_env_visualization.py` 末尾添一个 `def test_viz_xxx(env):` 即可，格式见现有例子。
共用 fixture `env` 是模块级 K=1 开放场地 env，覆盖 `env.start_states`
`env.target_states` `env.forbidden_set` 来定制，末尾 `save_trajectory_fig` 落盘即可。

---

## 常见问题

**Q: 跑出来提示 `No module named 'pytest'` 或 `pytest_html`**
A: `.venv/Scripts/python -m pip install pytest pytest-html`

**Q: 看不到 [PERF] 数字**
A: `run_tests.py` 已经带 `-s`；若手动 `pytest` 请加 `-s`（否则输出被捕获）。

**Q: 可视化 PNG 打开是空的 / 位置不对**
A: 检查测试 docstring 里预期终点；也有可能 YAML 的 `action_directions` 顺序被改过，
常量 `DIR_UP=3` / `DIR_DOWN=1` 等会失效。见 `_helpers.py` 顶部注释。

**Q: 性能测试在慢机器上跑不过断言**
A: 改 `test_env_performance.py` 顶部的阈值常量。
