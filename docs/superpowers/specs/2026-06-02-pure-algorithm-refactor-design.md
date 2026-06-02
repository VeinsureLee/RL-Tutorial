# 项目重构设计：从毕设RL平台到纯算法学习项目

**日期**：2026-06-02
**作者**：Veinsure Lee
**状态**：草案

---

## 1. 背景与目标

### 1.1 现状

本项目原为本科毕设"多机器人通信-感知协同导航与轨迹规划"，包含：

- 复杂的无线电通信模块（NOMA + 块对角化 + SIC + URLLC BER）
- 自研的 120×60 稀疏障碍物网格环境
- 7 个 RL 算法（DQN、MADQN、SharedMADQN、QMIX、VDN、PPO、MAPPO）
- 多个 YAML 配置文件（channel.yml、map.yml、env.yml、rl.yml、random_seed.yml）
- 通过 `sys.path.insert()` hack 解决 src/ 布局导入问题

### 1.2 重构目标

1. **转型为纯算法研究项目**，方便初学者学习多智能体强化学习
2. **彻底移除通信/无线电模块**及所有相关代码、配置、文档
3. **替换环境实现**：从自研稀疏障碍物 grid 转为基于 MiniGrid 的室内场景（房间、门、墙）
4. **配置驱动**：所有变量通过单一 `config.yml` 控制，去掉交互式智能体放置
5. **预留服务器部署能力**：通过 FastAPI 暴露算法接口，方便未来对接前端网站
6. **代码精简**：使用 `pyproject.toml` 替代 sys.path hack，统一接口降低耦合

---

## 2. 整体架构

### 2.1 目录结构

```
marl-nav/                          # 项目根
├── src/
│   ├── algorithms/
│   │   ├── __init__.py            # ALGORITHM_REGISTRY + build_algorithm()
│   │   ├── base.py                # BaseAlgorithm 抽象基类
│   │   ├── value_based/
│   │   │   ├── dqn/               # algo.py + qnet.py
│   │   │   ├── madqn/
│   │   │   ├── shared_madqn/
│   │   │   ├── qmix/              # algo.py + qnet.py + mixer.py
│   │   │   └── vdn/
│   │   └── policy_based/
│   │       ├── ppo/               # algo.py + net.py
│   │       └── mappo/
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── indoor_env.py          # 多智能体 MiniGrid wrapper
│   │   └── map_builder.py         # 从 yml 读取地图定义并构建
│   ├── core/
│   │   ├── trainer.py             # 统一 train() 循环
│   │   ├── tester.py              # 统一 test() + 可视化
│   │   ├── replay.py              # ReplayBuffer / JointReplayBuffer
│   │   └── plot.py                # 训练曲线绘图
│   ├── api/
│   │   ├── main.py                # FastAPI app 入口
│   │   ├── routes/
│   │   │   ├── train.py           # POST /train
│   │   │   ├── test.py            # POST /test
│   │   │   └── status.py          # GET /status/{run_id}
│   │   └── schemas.py             # Pydantic 请求/响应模型
│   └── utils/
│       ├── config.py              # 单一 YAML 加载器
│       ├── logger.py
│       └── paths.py
├── config/
│   └── config.yml                 # 唯一配置文件
├── maps/
│   └── default.yml                # 地图定义（房间/门/墙/起点/终点）
├── experiments/                   # 训练产出（自动生成）
├── tests/
├── main.py                        # CLI 入口
├── pyproject.toml                 # 包声明，替代 sys.path hack
└── requirements.txt
```

### 2.2 数据流

```
CLI 入口 (main.py) ──┐
                     ├──► build_algorithm() ──► core/trainer.py ──► experiments/
API 入口 (api/) ─────┘                      └──► core/tester.py
```

无论 CLI 还是 API，都通过 `build_algorithm()` 工厂构造算法，然后调用 `core/` 中的统一逻辑，保证两条入口行为一致。

---

## 3. 环境层设计（`src/envs/`）

### 3.1 MiniGrid 多智能体 Wrapper

MiniGrid 原生为单智能体，需要包装为多智能体 Gymnasium 接口：

```python
class IndoorEnv(gym.Env):
    def reset(self) -> dict[int, np.ndarray]:
        """返回 {agent_id: obs} 字典"""

    def step(self, actions: dict[int, int]) -> tuple[
        dict[int, np.ndarray],   # next_obs
        dict[int, float],        # rewards
        dict[int, bool],         # dones
        dict[int, dict],         # infos
    ]: ...
```

### 3.2 观测空间（config 可切换）

```yaml
env:
  observation_mode: "partial"   # partial | full
  partial_view_size: 7          # partial 模式视野大小（奇数）
```

- `partial`：每个智能体看到以自身为中心的 N×N 网格 → flatten 为向量输入
- `full`：整张地图 one-hot 编码，所有智能体位置可见

### 3.3 动作空间

固定 4 个离散动作：`{0: 上, 1: 下, 2: 左, 3: 右}`。移除原来的"停留"和"功率等级"。

### 3.4 奖励结构（config 可切换）

```yaml
env:
  reward_mode: "independent"    # independent | cooperative
  reward_goal: 10.0
  reward_step: -0.01
  reward_collision: -1.0        # 撞墙
  reward_team_bonus: 5.0        # 仅 cooperative：全员到达额外奖励
```

- `independent`：每个智能体只关心自己是否到达目标
- `cooperative`：在独立奖励基础上，加上团队 bonus（全员到达才发放）

### 3.5 地图定义（`maps/default.yml`）

```yaml
map:
  size: [15, 15]
  num_agents: 2
  num_goals: 2

  rooms:
    - id: room_1
      top_left: [0, 0]
      size: [7, 7]
    - id: room_2
      top_left: [0, 8]
      size: [7, 7]
    - id: room_3
      top_left: [8, 0]
      size: [15, 15]

  doors:
    - position: [3, 7]
    - position: [7, 4]

  agents_start: [[1, 1], [1, 13]]
  goals:        [[13, 1], [13, 13]]
```

`map_builder.py` 负责读取该 yml 并构建对应 MiniGrid 对象（基于 `RoomGrid` 或 `MultiRoom`）。

---

## 4. 算法层设计（`src/algorithms/`）

### 4.1 抽象基类

```python
# base.py
class BaseAlgorithm(ABC):
    @abstractmethod
    def take_action(self, states, explore: bool = True) -> list[int]: ...

    @abstractmethod
    def update(self, *args, **kwargs) -> dict: ...

    @abstractmethod
    def save(self, path: str): ...

    @abstractmethod
    def load(self, path: str): ...
```

### 4.2 算法分类与注册

```
algorithms/
├── base.py
├── value_based/
│   ├── dqn/        → 单智能体 Q-learning，入门首选
│   ├── madqn/      → 独立多智能体 Q-learning（DTDE）
│   ├── shared_madqn/ → 参数共享 MADQN
│   ├── qmix/       → CTDE，monotonic mixer
│   └── vdn/        → CTDE，加性值分解
└── policy_based/
    ├── ppo/        → 单智能体策略梯度
    └── mappo/      → 多智能体 PPO（CTDE）
```

### 4.3 统一注册表

```python
# algorithms/__init__.py
ALGORITHM_REGISTRY = {
    "dqn":          DQN,
    "madqn":        MADQN,
    "shared_madqn": SharedMADQN,
    "qmix":         QMIX,
    "vdn":          VDN,
    "ppo":          PPO,
    "mappo":        MAPPO,
}

def build_algorithm(name: str, env, cfg) -> BaseAlgorithm:
    return ALGORITHM_REGISTRY[name](env, cfg)
```

**关键改进**：新增算法只需添加子文件夹 + 在注册表加一行，**无需修改 trainer.py**，解决原项目中 trainer 与算法的紧耦合。

### 4.4 配置示例

```yaml
algorithm:
  name: "madqn"
  lr: 1e-4
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

  # on-policy 专用
  update_interval: 1024
  clip_epsilon: 0.2
  ppo_epochs: 5
  entropy_coef: 0.01
  gae_lambda: 0.95
```

---

## 5. Core 层设计（`src/core/`）

### 5.1 trainer.py

```python
def train(algo: BaseAlgorithm, env: IndoorEnv, cfg) -> TrainResult:
    """统一训练循环，根据 algo 类型自动区分 on/off-policy"""
```

`TrainResult` 是结构化数据类（dataclass 或 Pydantic 模型），包含：
- `history`：每个 episode 的 reward、steps 序列
- `model_path`：模型保存路径
- `run_id`：本次训练唯一标识

### 5.2 tester.py

```python
def test(algo: BaseAlgorithm, env: IndoorEnv, cfg) -> TestResult:
    """单次测试 + 生成可视化"""
```

`TestResult` 包含：`success`、`steps`、`total_reward`、`gif_path`、`png_path`。

### 5.3 replay.py

保留两种 buffer，删除原来的 `TargetReplayBuffer`（不再需要存通信目标）：
- `ReplayBuffer`：标准经验回放
- `JointReplayBuffer`：QMIX/VDN 用，存联合动作

### 5.4 plot.py

只保留两条核心曲线（删除原来的 4 路奖励分解和 BER 曲线）：
- `episode_reward`
- `steps_to_goal`

---

## 6. API 层设计（`src/api/`）

### 6.1 端点定义

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/train` | 提交训练任务，立即返回 `run_id` |
| `GET` | `/status/{run_id}` | 查询训练进度 |
| `POST` | `/test` | 用已有模型跑测试，返回 gif/指标 |

### 6.2 异步设计

训练是长任务，使用 FastAPI 的 `BackgroundTasks` 或 `asyncio.create_task`：
- `POST /train` → 启动后台任务 → 立即返回 `{"run_id": "..."}`
- 前端轮询 `GET /status/{run_id}` 获取进度
- 训练完成后状态变为 `completed`，可下载模型路径

### 6.3 Pydantic Schema

```python
class TrainRequest(BaseModel):
    algorithm: str                # "dqn" | "madqn" | ...
    map_file: str = "default"     # 读 maps/{map_file}.yml
    config_overrides: dict = {}   # 覆盖 config.yml 字段

class StatusResponse(BaseModel):
    run_id: str
    status: str                   # "running" | "completed" | "failed"
    episode: int
    total_episodes: int
    latest_reward: float
    model_path: str | None
```

### 6.4 部署友好性

- 单一 `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` 即可启动
- 训练产物全部落在 `experiments/` 下，前端通过 API 读取
- 无状态设计：所有训练状态保存在文件系统，便于重启恢复

---

## 7. 配置系统

### 7.1 单一配置文件

`config/config.yml` 是唯一入口，结构如下：

```yaml
seed: 42

env:
  map_file: "default"
  observation_mode: "partial"
  partial_view_size: 7
  reward_mode: "independent"
  reward_goal: 10.0
  reward_step: -0.01
  reward_collision: -1.0
  reward_team_bonus: 5.0

algorithm:
  name: "madqn"
  lr: 1e-4
  # ... 其他超参

logging:
  log_dir: "experiments"
  save_interval: 50
```

### 7.2 配置加载

`utils/config.py` 提供：
- `load_config(path: str) -> dict`：加载 YAML
- `merge_overrides(cfg: dict, overrides: dict) -> dict`：CLI 或 API 覆盖

去掉原项目的 `{value, description}` 包裹结构，直接用扁平 YAML。

---

## 8. 迁移策略

### 8.1 删除清单

| 模块 | 操作 |
|------|------|
| `src/communication/`（整个） | **删除** |
| `config/base/channel.yml` | **删除** |
| `config/base/map.yml`（天线、障碍物字段） | **删除** |
| `config/dynamic/`（scenario.npz、radio_map_cache.npz） | **删除** |
| `src/config/generator/` | **删除** |
| `src/config/customer_choice.py`（交互式选点） | **删除** |
| `src/env/env.py`（自研 grid env） | **替换为 MiniGrid wrapper** |
| 现有奖励中的 `comm_rewards`、`ber_*` | **删除** |
| `rl_algorithms/trainer.py` 中的通信日志、BER 跟踪 | **简化** |
| `requirements.txt`：`pygame` | **删除** |
| `requirements.txt`：增加 `minigrid`、`gymnasium`、`fastapi`、`uvicorn`、`pydantic` | **新增** |
| `models/` 旧 checkpoint | **删除** |
| `tests/test_communication.py`、`test_ofdma.py` | **删除** |
| `docs/` 下毕设论文相关文档 | **删除** |

### 8.2 保留并改造清单

| 模块 | 操作 |
|------|------|
| 7 个 RL 算法核心逻辑 | **保留**，按 value-based/policy-based 重组目录 |
| `ReplayBuffer`、`JointReplayBuffer` | **保留**，删除 `TargetReplayBuffer` |
| 训练曲线绘图 | **简化**为 2 条曲线 |
| 日志/路径工具 | **保留**，迁入 `src/utils/` |

### 8.3 新增清单

- `src/envs/indoor_env.py`、`map_builder.py`
- `src/algorithms/base.py`
- `src/api/` 整个目录
- `maps/default.yml`
- `pyproject.toml`
- `config/config.yml`（合并替代多个 yml）

---

## 9. 测试策略

- `tests/test_envs.py`：MiniGrid wrapper 的 reset/step 接口正确性
- `tests/test_algorithms.py`：每个算法跑 10 episode 烟雾测试
- `tests/test_api.py`：FastAPI 端点用 `TestClient` 测试
- 不引入 CI，本地 `pytest` 跑通即可

---

## 10. 依赖变更

### 新增

```
minigrid >= 2.3.0
gymnasium >= 0.29
fastapi >= 0.110
uvicorn >= 0.27
pydantic >= 2.0
```

### 移除

```
pygame
```

### 保留

```
torch >= 2.0
numpy
matplotlib
seaborn
pandas
pillow
pyyaml
```

---

## 11. 验收标准

1. `pip install -e .` 可成功安装，无需 sys.path hack
2. `python main.py --algo dqn --mode train` 可在 MiniGrid 室内环境中训练
3. `python main.py --algo madqn --mode test --model_path ...` 可生成 gif/png 可视化
4. `uvicorn src.api.main:app` 启动后，可通过 HTTP 触发训练并查询状态
5. 所有 7 个算法在新环境下能跑通（不要求收敛到最优，但接口必须工作）
6. `pytest tests/` 全部通过
7. 项目根目录无 `communication/`、`channel.yml`、`scenario.npz` 等通信遗留
