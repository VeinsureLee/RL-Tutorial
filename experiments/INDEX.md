# Experiments Index

> 本目录存放**所有"跑出来的东西"**：训练 run、单元测试报告、跨 run 对比。
> Ctrl+F 搜 `algo / K / ω / lr / 日期` 可快速定位。

## 目录

| 路径 | 作用 |
| --- | --- |
| [`runs/`](runs/) | 每次训练 = 一个自包含文件夹（log + config + metrics + figs + test 产物） |
| [`comparisons/`](comparisons/) | 跨 run 对比（ω 扫描 / agent scaling / NOMA vs OFDMA …） |
| [`unit_tests/`](unit_tests/) | pytest 报告按日期归档 |

---

## Runs 索引

| 日期 | Run 名 | 算法 | K | ω | lr | 步数 | Return | -log₁₀BER | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-04-23 | [madqn_K1_omega1_lr1e-04_smoke](runs/20260423_2233_madqn_K1_omega1_lr1e-04_smoke/) | MADQN | 1 | 1.0 | 1e-4 | 20 | -22.0 | 20.0 | smoke | 重构后首次端到端验证 |

> 旧 run 归档在 [`../archive/legacy_logs/`](../archive/legacy_logs/) 与 [`../archive/legacy_results/`](../archive/legacy_results/)。

---

## 单元测试索引

| 日期 | 报告 | 通过 / 总数 |
|---|---|---|
| 2026-04-23 | [unit_tests/20260423/](unit_tests/20260423/) | 25 / 25 ✓ |

---

## Comparisons 索引

| 名字 | 涉及 runs | 主图 | 结论 |
|---|---|---|---|
| — | — | — | 待跑 |

---

## Run 目录命名规范

```
<YYYYMMDD>_<HHMM>_<algo>_K<N>_<关键超参>[_<tag>]
```

由 `src/utils/run_manager.RunContext.new(...)` 自动生成。示例：

- `20260425_1030_madqn_K4_omega1_lr1e-04`           — baseline
- `20260425_1200_ddqn_K4_omega1_lr1e-04`            — 算法替换实验
- `20260425_1400_madqn_K4_omega0_lr1e-04_nonoma`    — 关通信消融
- `20260425_1500_madqn_K4_omega1_lr1e-04_rerun2`    — 复现跑

## Run 目录结构

每次运行自动生成：

```
<run_name>/
├── summary.md              ★ 本次意图 + 结论（骨架自动生成，意图与结论由人手填）
├── config.snapshot.yml     用过的 YAML 全量快照（env_cfg + rl_cfg）
├── run.log                 训练日志
├── metrics.csv             trainer.history 全量（return / step / approach / comm / ber + 每 agent）
├── model.pth               权重
├── figs/                   训练曲线
│   ├── return.png
│   ├── ber.png
│   ├── step_reward.png
│   ├── approach_reward.png
│   └── ber_reward.png
└── test/                   同一 run 的测试产物（自动在 train 结束后跑一次）
    ├── nav.gif / signal.gif
    └── nav.png / signal.png
```
