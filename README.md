# MARL-Nav

> 面向初学者的多智能体强化学习导航研究平台

七种经典 RL 算法（DQN / MADQN / SharedMADQN / VDN / QMIX / PPO / MAPPO）统一跑在可配置的室内格子世界环境里（房间 + 门 + 墙）。代码结构清晰、一键训练、自带 FastAPI 接口，方便未来对接网站。

---

## 效果预览

地图默认为 25×25、四房间布局，两个智能体从对角出发，穿过两道门到达对角目标：

![default map](maps/preview/default.png)

生成地图预览：`python scripts/visualize_map.py --map default --scale 40`

---

## 系统架构

架构图位于 `docs/`，用 [draw.io / diagrams.net](https://app.diagrams.net/) 打开：

| 文件 | 内容 |
|------|------|
| `docs/architecture.drawio` | 整体分层架构（入口 → 配置 → 环境 → 算法 → Core → 输出） |
| `docs/training_flow.drawio` | 训练流程图（off-policy / on-policy 两条分支） |
| `docs/algorithms.drawio` | 算法分类树（Value-based / Policy-based + buffer 类型对照） |

```
入口层        main.py (CLI)          api/main.py (FastAPI)
                       │                    │
配置层     config/config.yml    maps/default.yml    utils/config.py
                       │
环境层     envs/map_builder.py  ──►  envs/indoor_env.py (IndoorEnv)
                       │
算法层     algorithms/ALGORITHM_REGISTRY[name](env, cfg)
           ├── value_based: DQN  MADQN  SharedMADQN  VDN  QMIX
           └── policy_based: PPO  MAPPO
                       │
Core 层    core/trainer.py  core/tester.py  core/replay.py  core/plot.py
                       │
输出       experiments/<run>/model.pth  figs/  nav.gif  run.log
```

---

## 快速开始

```bash
# 1. 安装（editable 模式，不需要手动加 sys.path）
pip install -e .

# 2. 训练
python main.py --algo dqn   --mode train
python main.py --algo madqn --mode train --num_episodes 500 --lr 5e-5
python main.py --algo qmix  --mode train --map default --tag exp1

# 3. 测试（加载已训模型）
python main.py --algo madqn --mode test \
    --model_path experiments/<run_dir>/model.pth --max_steps 500

# 4. 可视化训练地图
python scripts/visualize_map.py --map default --scale 40

# 5. 启动 API 服务器
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 6. 跑所有测试
pytest tests/
```

CLI 参数优先级：**命令行 > config/config.yml > 代码默认值**

---

## 环境说明

### 地图（`maps/default.yml`）

```
┌─────────────────┬─────────────────┐
│                 │                 │
│   TL 房间       │   TR 房间       │
│   A0 起点       │       A1 起点   │
│                 │                 │
└────┬──────── 门 ┴──────── 门 ─────┘
     门                   门
┌────┴────────────────────┴──────────┐
│                                    │
│          Bottom 大房间              │
│  G1 目标                  G0 目标  │
│                                    │
└────────────────────────────────────┘
```

A0 的目标 G0 在对角，A1 的目标 G1 也在对角——每个智能体至少要穿越两扇门。

### 地图 yml 格式

```yaml
map:
  size: [25, 25]          # [行, 列]
  num_agents: 2
  num_goals: 2
  rooms:
    - id: room_tl
      top_left: [0, 0]
      size: [13, 13]      # 高×宽（包含边界墙）
  doors:
    - position: [6, 12]   # [行, 列]，必须在共享墙上
  agents_start: [[2, 2], [2, 22]]
  goals:        [[22, 22], [22, 2]]
```

### 观测空间

| 模式 | 说明 | 维度 |
|------|------|------|
| `partial` | 以智能体为中心的 7×7 视野，flatten | 49 |
| `full`    | 整张 25×25 地图编码，flatten | 625 |

通过 `config.yml` 中的 `env.observation_mode` 切换。

### 奖励结构

| 模式 | 说明 |
|------|------|
| `independent` | 每个 agent 只关心自己是否到达目标 |
| `cooperative` | 独立奖励 + 全员到达时的团队 bonus |

---

## 算法说明

### 分类概览

```
强化学习算法
├── Value-based（基于值函数，off-policy）
│   ├── DTDE（去中心化训练+执行）
│   │   ├── DQN          单智能体 Q-learning，入门首选
│   │   ├── MADQN        每 agent 独立 Q 网络，无协调
│   │   └── SharedMADQN  所有 agent 共享同一 Q 网络参数
│   └── CTDE（中心化训练，去中心化执行）
│       ├── VDN          Q_tot = Σ Q_i，最简单的值分解
│       └── QMIX         Q_tot 由单调 mixer 生成，更强表达力
└── Policy-based（基于策略，on-policy）
    ├── PPO    单智能体 Actor-Critic + clip surrogate
    └── MAPPO  多智能体 PPO，共享 actor + 中心化全局 critic
```

### 算法对比表

| 算法 | 智能体数 | 范式 | Buffer 类型 | 网络参数 | 核心特点 |
|------|----------|------|-------------|----------|----------|
| DQN | 单 (agent 0) | DTDE | `single` | 独立 | 最基础 Q-learning |
| MADQN | 多 | DTDE | `per_agent` | 独立 N 份 | 无协调，最简单多智能体 |
| SharedMADQN | 多 | DTDE | `per_agent` | 1 份共享 | 同质 agent 的隐式数据增强 |
| VDN | 多 | CTDE | `joint` | 独立 N 份 | Q_tot = Σ Q_i，IGM 满足加性 |
| QMIX | 多 | CTDE | `joint` | Q_i + mixer | 单调性约束，比 VDN 表达更强 |
| PPO | 单 (agent 0) | — | `none` | Actor-Critic | on-policy，clip 保证稳定 |
| MAPPO | 多 | CTDE | `none` | 共享 actor + Critic | CTDE on-policy |

### Buffer 类型说明

- **`single`** — 标准 `ReplayBuffer`，只存 agent 0 的 transition
- **`per_agent`** — `{i: ReplayBuffer}` 字典，每 agent 各一份
- **`joint`** — `JointReplayBuffer`，每条存 N 个 agent 的联合 transition，shape `(B, N, *)`
- **`none`** — on-policy rollout，不需要经验回放

---

## 训练流程

```
main.py
  │
  ├─ load_config(config/config.yml) + merge CLI overrides
  │
  ├─ IndoorEnv(cfg)           构建多智能体室内环境
  │     └─ map_builder → 15×15/25×25 numpy 网格
  │
  ├─ build_algorithm(name, env, cfg)    从注册表实例化算法
  │
  └─ core/trainer.train(algo, env, cfg, run_dir)
        │
        ├─ [off-policy]  ReplayBuffer → epsilon-greedy → TD 更新
        │                每 update_freq 步同步 target 网络
        │
        └─ [on-policy]   rollout 收集 → GAE → PPO clip 更新
                         每 update_interval 步触发一次更新
```

训练产物自动落到 `experiments/<YYYYMMDD_HHMMSS>_<algo>[_<tag>]/`：

```
experiments/20260602_175039_dqn_baseline/
├── model.pth          模型权重
├── run.log            训练日志
└── figs/
    ├── reward.png     每 episode 总奖励曲线
    └── steps.png      每 episode 步数曲线
```

---

## API 服务

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/train` | 提交训练任务，立即返回 `run_id` |
| `GET`  | `/status/{run_id}` | 查询进度（episode / status / reward） |
| `POST` | `/test` | 用已有模型跑一次测试，返回 gif/指标 |
| `GET`  | `/health` | 健康检查 |

训练在后台 asyncio 任务中执行，前端轮询 `/status/{run_id}`。

**v1 限制**：服务重启后 run 状态丢失；进度只在训练结束后更新（无实时 episode 计数）。

---

## 配置参考

所有变量都在 `config/config.yml`，CLI 参数覆盖对应字段：

```yaml
seed: 42

env:
  map_file: "default"          # 读 maps/default.yml
  observation_mode: "partial"  # partial | full
  partial_view_size: 7         # partial 模式视野（奇数）
  reward_mode: "independent"   # independent | cooperative
  reward_goal: 10.0
  reward_step: -0.01
  reward_collision: -1.0
  reward_team_bonus: 5.0       # 仅 cooperative 模式

algorithm:
  name: "dqn"
  lr: 1.0e-4
  gamma: 0.99
  hidden_dim: 128
  num_episodes: 500
  episode_length: 500
  batch_size: 64
  # off-policy 专用
  epsilon: 0.9
  epsilon_min: 0.05
  epsilon_decay: 0.995
  replay_buffer_size: 50000
  train_interval: 4
  update_freq: 10
  # on-policy 专用 (PPO/MAPPO)
  update_interval: 1024
  clip_epsilon: 0.2
  ppo_epochs: 5
  entropy_coef: 0.01
  gae_lambda: 0.95
  value_coef: 0.5

logging:
  log_dir: "experiments"
  save_interval: 50
```

---

## 扩展指南

### 新增算法

1. 在 `src/algorithms/{value_based|policy_based}/<name>/` 下创建 `algo.py`、`qnet.py`（或 `net.py`）、`__init__.py`
2. 继承 `algorithms.base.BaseAlgorithm`，实现 `take_action / update / save / load`
3. 覆盖 `required_buffer()` 返回 `"single" | "per_agent" | "joint" | "none"`
4. 若为 on-policy，覆盖 `is_on_policy = True`，并在 `core/trainer.py` 的 `_train_on_policy()` 中添加分支
5. 在 `src/algorithms/__init__.py` 的 `ALGORITHM_REGISTRY` 里加一行

### 新增地图

在 `maps/<name>.yml` 中定义房间/门/起点/终点，用 `--map <name>` 选择。
用 `python scripts/visualize_map.py --map <name>` 验证地图连通性。

---

## 目录结构

```
marl-nav/
├── src/
│   ├── algorithms/
│   │   ├── base.py                   BaseAlgorithm 抽象基类
│   │   ├── __init__.py               ALGORITHM_REGISTRY + build_algorithm()
│   │   ├── value_based/
│   │   │   ├── dqn/                  algo.py  qnet.py
│   │   │   ├── madqn/
│   │   │   ├── shared_madqn/
│   │   │   ├── vdn/
│   │   │   └── qmix/                 algo.py  qnet.py  mixer.py
│   │   └── policy_based/
│   │       ├── ppo/                  algo.py  net.py
│   │       └── mappo/
│   ├── envs/
│   │   ├── indoor_env.py             IndoorEnv（多智能体 Gymnasium 接口）
│   │   └── map_builder.py            yml → numpy 网格
│   ├── core/
│   │   ├── trainer.py                统一 train()
│   │   ├── tester.py                 统一 test() + gif/png
│   │   ├── replay.py                 ReplayBuffer  JointReplayBuffer
│   │   └── plot.py                   训练曲线
│   ├── api/
│   │   ├── main.py                   FastAPI app
│   │   ├── schemas.py                Pydantic 请求/响应模型
│   │   ├── runs.py                   RunState 注册表 + 异步训练
│   │   └── routes/                   train.py  status.py  test.py
│   └── utils/
│       ├── config.py                 load_config + merge_overrides
│       ├── logger.py                 get_logger
│       └── paths.py                  project_root + map_path + experiments_dir
├── config/
│   └── config.yml                    唯一配置文件
├── maps/
│   ├── default.yml                   25×25 四房间默认地图
│   └── preview/default.png           地图渲染预览
├── docs/
│   ├── architecture.drawio           系统分层架构图
│   ├── training_flow.drawio          训练流程图
│   └── algorithms.drawio             算法分类树
├── scripts/
│   └── visualize_map.py              地图可视化工具
├── experiments/                      训练产物（gitignore）
├── tests/                            pytest 测试
├── main.py                           CLI 入口
└── pyproject.toml                    包声明
```

---

## 测试

```bash
pytest tests/ -v
```

测试覆盖：config 加载、路径工具、地图解析、环境接口、经验回放、FastAPI 端点。

---

## 语言约定

- **代码注释 / docstring**：中文
- **CLI 输出 / 日志 / matplotlib 标签**：ASCII 英文（避免 Windows cmd 乱码）
- `matplotlib.rcParams["axes.unicode_minus"] = False` 已在 `core/plot.py` 全局设置
