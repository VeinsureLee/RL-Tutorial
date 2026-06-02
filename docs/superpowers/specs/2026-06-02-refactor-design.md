# Refactor Design: MARL Navigation Research Platform

**Date:** 2026-06-02  
**Status:** Approved  
**Scope:** Full project refactor — from RL graduation thesis to beginner-friendly algorithm research platform

---

## 1. Goals

- Remove all wireless communication (BER/NOMA/SIC) modules entirely
- Replace sparse-obstacle grid env with MiniGrid-based indoor env (rooms, doors, walls)
- Reorganize 7 RL algorithms into value-based / policy-based categories
- Single `config.yml` drives all settings; no interactive placement tools
- FastAPI interface for future web deployment (async train + status polling)
- Eliminate `sys.path` hacks via `pyproject.toml` src-layout packaging
- Code readable by beginners: clear layers, minimal abstraction

---

## 2. Directory Structure

```
marl-nav/
├── src/
│   ├── algorithms/
│   │   ├── __init__.py            # ALGORITHM_REGISTRY + build_algorithm()
│   │   ├── base.py                # BaseAlgorithm abstract class
│   │   ├── value_based/
│   │   │   ├── dqn/               # algo.py, qnet.py
│   │   │   ├── madqn/             # algo.py, qnet.py
│   │   │   ├── shared_madqn/      # algo.py, qnet.py
│   │   │   ├── qmix/              # algo.py, qnet.py, mixer.py
│   │   │   └── vdn/               # algo.py, qnet.py
│   │   └── policy_based/
│   │       ├── ppo/               # algo.py, net.py
│   │       └── mappo/             # algo.py, net.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── indoor_env.py          # Multi-agent MiniGrid wrapper (Gymnasium)
│   │   └── map_builder.py         # Reads maps/*.yml, builds MiniGrid object
│   ├── core/
│   │   ├── trainer.py             # Unified train() — on/off-policy
│   │   ├── tester.py              # Unified test() + nav.gif/png output
│   │   ├── replay.py              # ReplayBuffer, JointReplayBuffer
│   │   └── plot.py                # episode_reward + steps_to_goal curves
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   ├── routes/
│   │   │   ├── train.py           # POST /train
│   │   │   ├── test.py            # POST /test
│   │   │   └── status.py          # GET /status/{run_id}
│   │   └── schemas.py             # Pydantic request/response models
│   └── utils/
│       ├── config.py              # Single YAML loader (config + map)
│       ├── logger.py
│       └── paths.py
├── config/
│   └── config.yml                 # Single config file
├── maps/
│   └── default.yml                # Map definition (rooms/doors/agents/goals)
├── experiments/                   # Auto-generated training outputs
│   └── {run_id}/
│       ├── model.pth
│       ├── metrics.csv
│       ├── figs/
│       └── test/
├── tests/
├── main.py                        # CLI entry point
├── pyproject.toml                 # src-layout package declaration
└── requirements.txt
```

---

## 3. Environment Layer

### `src/envs/indoor_env.py`

Multi-agent wrapper around MiniGrid. Exposes Gymnasium-compatible interface:

```python
class IndoorEnv(gym.Env):
    def reset() -> dict[str, obs]
    def step(actions: dict[str, int]) -> tuple[obs, rewards, dones, infos]
```

**Action space:** 4 discrete actions (up / down / left / right). No power levels, no STAY.

**Observation space** (config-switchable):
- `partial`: N×N view centered on agent, flattened. N set by `env.partial_view_size`.
- `full`: One-hot encoding of entire map including all agent positions.

**Reward structure** (config-switchable):
- `independent` mode: each agent rewarded only for its own goal arrival
- `cooperative` mode: independent rewards + team bonus when all agents arrive

Reward components:
- `reward_goal`: on reaching goal
- `reward_step`: constant time penalty per step
- `reward_team_bonus`: (cooperative only) all agents reached their goals

### `src/envs/map_builder.py`

Reads `maps/*.yml` and constructs the corresponding MiniGrid environment. Uses MiniGrid's `RoomGrid` / `MultiRoom` primitives.

### `maps/default.yml` format

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

---

## 4. Algorithm Layer

### `src/algorithms/base.py`

```python
class BaseAlgorithm(ABC):
    @abstractmethod
    def take_action(self, states, explore=True) -> list[int]: ...
    @abstractmethod
    def update(self, *args, **kwargs) -> dict: ...
    @abstractmethod
    def save(self, path: str): ...
    @abstractmethod
    def load(self, path: str): ...
```

### Registry (`src/algorithms/__init__.py`)

```python
ALGORITHM_REGISTRY = {
    "dqn": DQN, "madqn": MADQN, "shared_madqn": SharedMADQN,
    "qmix": QMIX, "vdn": VDN, "ppo": PPO, "mappo": MAPPO,
}

def build_algorithm(name: str, env, cfg) -> BaseAlgorithm:
    return ALGORITHM_REGISTRY[name](env, cfg)
```

Adding a new algorithm = new subfolder + one line in registry. No changes to `trainer.py`.

### Classification

| Category | Algorithms |
|----------|-----------|
| Value-based (DTDE) | DQN, MADQN, SharedMADQN |
| Value-based (CTDE) | QMIX, VDN |
| Policy-based | PPO, MAPPO |

### `config.yml` algorithm section

```yaml
algorithm:
  name: "madqn"
  lr: 1e-4
  gamma: 0.99
  hidden_dim: 128
  num_episodes: 500
  episode_length: 500
  batch_size: 64
  # off-policy
  epsilon: 0.9
  epsilon_min: 0.05
  epsilon_decay: 0.995
  replay_buffer_size: 50000
  train_interval: 4
  update_freq: 10
  # on-policy (PPO/MAPPO)
  update_interval: 1024
  clip_epsilon: 0.2
  ppo_epochs: 5
  entropy_coef: 0.01
  gae_lambda: 0.95
```

---

## 5. Core Layer

### `src/core/trainer.py`

```python
def train(algo: BaseAlgorithm, env: IndoorEnv, cfg) -> TrainResult:
    ...
# TrainResult: { history: list[dict], model_path: str, run_id: str }
```

Detects on-policy vs off-policy from algorithm type. Saves `metrics.csv` and model checkpoint.

### `src/core/tester.py`

```python
def test(algo: BaseAlgorithm, env: IndoorEnv, cfg) -> TestResult:
    ...
# TestResult: { success: bool, steps: int, total_reward: float,
#               gif_path: str, png_path: str }
```

Outputs `nav.gif` and `nav.png` only (no signal heatmap).

### `src/core/replay.py`

- `ReplayBuffer`: single-agent off-policy
- `JointReplayBuffer`: QMIX/VDN (stores joint actions + global state)

Communication-related fields removed.

### `src/core/plot.py`

Two curves only: `episode_reward`, `steps_to_goal`.

---

## 6. API Layer

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/train` | Submit training job, returns `run_id` immediately |
| `GET`  | `/status/{run_id}` | Poll training progress |
| `POST` | `/test` | Run one test episode with saved model |

### Async design

Training runs as a background `asyncio` task. Client polls `/status/{run_id}`.

### Schemas

```python
class TrainRequest(BaseModel):
    algorithm: str
    map_file: str = "default"
    config_overrides: dict = {}

class StatusResponse(BaseModel):
    run_id: str
    status: str           # "running" | "completed" | "failed"
    episode: int
    total_episodes: int
    latest_reward: float
    model_path: str | None

class TestRequest(BaseModel):
    run_id: str
    max_steps: int = 500

class TestResult(BaseModel):
    success: bool
    steps: int
    total_reward: float
    gif_path: str
```

---

## 7. Config Structure

### `config/config.yml`

```yaml
env:
  observation_mode: "partial"     # partial | full
  partial_view_size: 7
  reward_mode: "independent"      # independent | cooperative
  reward_goal: 10.0
  reward_step: -0.01
  reward_team_bonus: 5.0

algorithm:
  name: "madqn"
  lr: 1e-4
  gamma: 0.99
  hidden_dim: 128
  num_episodes: 500
  episode_length: 500
  batch_size: 64
  epsilon: 0.9
  epsilon_min: 0.05
  epsilon_decay: 0.995
  replay_buffer_size: 50000
  train_interval: 4
  update_freq: 10
  update_interval: 1024
  clip_epsilon: 0.2
  ppo_epochs: 5
  entropy_coef: 0.01
  gae_lambda: 0.95

training:
  device: "auto"                  # auto | cpu | cuda
  seed: 42
  save_interval: 50               # save checkpoint every N episodes
```

---

## 8. Data Flow

```
CLI: python main.py --algo madqn --map default
         │
         ▼
    utils/config.py          Load config.yml + maps/default.yml
         │
         ▼
    envs/map_builder.py      Build MiniGrid map object
         │
         ▼
    envs/indoor_env.py       Wrap as multi-agent Gymnasium env
         │
         ▼
    algorithms/build_algorithm()   Instantiate from registry
         │
         ▼
    core/trainer.py          train() loop
         │
         ├── experiments/{run_id}/model.pth
         ├── experiments/{run_id}/metrics.csv
         └── experiments/{run_id}/figs/

API: POST /train → same pipeline, async background task, poll /status/{run_id}
```

---

## 9. Migration: Delete / Rewrite / Keep

| Module | Action | Notes |
|--------|--------|-------|
| `src/communication/` | **DELETE** | Entire wireless module removed |
| `config/base/channel.yml` | **DELETE** | |
| `config/base/map.yml` (antenna fields) | **DELETE** | Keep map_size, num_agents |
| `src/config/customer_choice.py` | **DELETE** | Interactive placement tool |
| `src/config/generator/` | **DELETE** | Replaced by map_builder.py |
| `env/env.py` BER/comm_reward/power_level | **DELETE** | |
| `rl_algorithms/` each algo core | **MIGRATE** → `src/algorithms/` | Standardize interface |
| `rl_algorithms/trainer.py` | **REWRITE** → `src/core/trainer.py` | Remove comm reward streams |
| `rl_algorithms/tester.py` | **REWRITE** → `src/core/tester.py` | Remove signal.gif |
| `rl_algorithms/replay.py` | **MIGRATE** → `src/core/replay.py` | Remove comm fields |
| `utils/` | **MIGRATE** → `src/utils/` | Keep logger, paths; remove config_handler |
| `tests/test_communication.py` | **DELETE** | |
| `tests/test_ofdma.py` | **DELETE** | |
| `main.py` | **REWRITE** | Simplified CLI, no sys.path hack |

---

## 10. Packaging (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "marl-nav"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

Install with `pip install -e .` — no `sys.path` manipulation anywhere.

---

## 11. New Dependencies

| Package | Purpose |
|---------|---------|
| `minigrid` | Indoor grid environment |
| `fastapi` | API layer |
| `uvicorn` | ASGI server for FastAPI |
| `pydantic` | Request/response validation |

Remove: `pygame` (no longer used for rendering).
