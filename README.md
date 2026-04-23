# 多机器人通感协同导航与轨迹规划

> 复现：《工业互联网场景下基于强化学习的多机器人协同控制与通信资源分配算法研究》（2024）

基于 120×60 栅格 + 深度强化学习（DQN / MADQN）的多机器人导航。每个 agent 同时选择
**移动方向**与**发射功率等级**；奖励由导航（路径接近 / 禁区 / 到达）与通信（NOMA+BD+SIC+URLLC BER）两部分合成。

---

## 快速开始

```bash
# 依赖
pip install -r requirements.txt

# 训练（产物自动落到 experiments/runs/<timestamp>_<algo>_K<N>_.../）
python main.py --model madqn --mode train
python main.py --model madqn --mode train --num_episodes 200 --lr 5e-5 --tag baseline

# 测试（加载已训模型，产物落到对应 run 目录的 test/）
python main.py --model madqn --mode test --model_path experiments/runs/<run_name>/model.pth

# 重新生成场景（config/dynamic/scenario.npz）
python -m config.generator.main

# 单元测试（HTML 报告落到 experiments/unit_tests/<date>/）
pytest --html=experiments/unit_tests/$(date +%Y%m%d)/pytest.html --self-contained-html
```

---

## 目录地图

```
Graduation_Project/
├── README.md                    ← 你在这里
├── main.py                      CLI 入口
├── requirements.txt
│
├── config/                      数据层（人可见、可编辑）
│   ├── base/*.yml               静态超参
│   └── dynamic/*.npz            运行时生成
│
├── src/                         ═══ 所有源代码 ═══
│   ├── config/                  配置中枢（yml_config + generator）
│   ├── env/                     MultiRobotEnv
│   ├── communication/           NOMA + BD + SIC + URLLC
│   ├── rl_algorithms/           DQN / MADQN / trainer / tester / plot
│   └── utils/                   path / logger / run_manager
│
├── tests/                       ═══ 单元测试 ═══
│
├── experiments/                 ═══ 所有"跑出来的东西" ═══
│   ├── INDEX.md                 ★ 训练 run 总索引
│   ├── runs/<run>/              每次训练 = 一个自包含目录
│   ├── comparisons/             跨 run 对比
│   └── unit_tests/<date>/       pytest HTML 报告
│
├── references/                  ═══ 参考文献 ═══
│   ├── INDEX.md                 ★ 编号 / 主题 / 引用位置
│   ├── reproduced/              复刻的论文
│   └── related/                 相关文献
│
├── paper/                       LaTeX 论文源
├── plans/                       进度跟踪（Student/Mentor/Paper/Project）
├── todo/                        实验待办
├── reports/                     开题 / 中期 / 验收报告
├── docs/                        系统设计图
└── archive/                     ═══ 重构前历史归档（只读） ═══
```

---

## 每个模块的 README

点下面的链接就能知道每个模块具体做什么：

| 模块 | README |
| --- | --- |
| 配置 | [`src/config/README.md`](src/config/README.md) |
| 环境 | [`src/env/README.md`](src/env/README.md) |
| 通信 | [`src/communication/README.md`](src/communication/README.md) |
| RL 算法 | （待补） |
| 工具 | （待补） |

---

## 查找"跑过的东西"

- **某次训练** → 去 [`experiments/INDEX.md`](experiments/INDEX.md) 的 runs 表，按 `algo / K / ω / lr / 日期` 搜
- **跨 run 对比** → [`experiments/comparisons/INDEX.md`](experiments/comparisons/INDEX.md)
- **某天的单元测试** → [`experiments/unit_tests/`](experiments/unit_tests/)
- **某篇参考文献** → [`references/INDEX.md`](references/INDEX.md)
- **旧的训练日志 / 结果** → [`archive/`](archive/)

每个 run 目录都是**自包含**的（见 `experiments/INDEX.md` 末尾的 run 目录规范）：
`summary.md`（意图 + 结论） + `config.snapshot.yml`（用过的 YAML） + `run.log` + `metrics.csv` + `model.pth` + `figs/` + `test/`。

---

## 语言约定

- **代码注释与 docstring**：中文
- **CLI 输出 / 日志 / matplotlib 标签**：ASCII 英文（避免 Windows `cmd` 乱码）
- `matplotlib.rcParams["axes.unicode_minus"] = False`（避免 `U+2212` 字体警告）

---

## 进度

见 [`plans/Project.md`](plans/Project.md) 与 [`experiments/INDEX.md`](experiments/INDEX.md) 最后一行。
