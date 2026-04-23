# Archive（只读历史归档）

> 本目录下的内容都是**重构前**的历史产物，保留仅作为历史追溯。
> 新的训练请使用 [`../experiments/`](../experiments/)；新的单元测试请使用 [`../tests/`](../tests/)。
> **不要**在这些目录里新增文件。

## 目录

| 路径 | 原始位置 | 内容 |
| --- | --- | --- |
| `legacy_logs/` | 原 `logs/` | 重构前的训练日志（扁平 `<algo>_<ts>.log`） |
| `legacy_results/` | 原 `results/` | 重构前的 `Train/` `compare/` `figs/` `gif/` `png/` 产物 |
| `legacy_models/` | 原 `models/` 中命名不规范的权重 | `madqn_model_test*.pth` 之类 |
| `legacy_experiments/` | 原 `实验规则/` | 旧实验规则文档与各 scan 结果 |

## 如果需要迁移到新体系

单个旧 run 搬到新 `experiments/runs/` 的做法：
1. 新建 `experiments/runs/<date>_<time>_<algo>_K<N>_<legacy>/`
2. 拷贝对应 `legacy_logs/<algo>_<ts>.log` 为 `run.log`
3. 拷贝 `legacy_results/figs/<prefix>_*.png` 到 `figs/`
4. 拷贝 `legacy_results/gif/<prefix>_*.gif` 到 `test/`
5. 手写一份 `summary.md`
