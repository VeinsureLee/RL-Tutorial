# Pure-Algorithm Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the multi-robot DRL graduation project into a beginner-friendly pure-algorithm research codebase: remove all wireless/communication code, replace the custom grid env with a MiniGrid indoor wrapper, organize the 7 RL algorithms under `value_based/` and `policy_based/`, add a FastAPI layer for future web deployment, and collapse the multi-yml config into one `config.yml` driven by a proper `src/` layout with `pyproject.toml`.

**Architecture:** Greenfield code under `src/{algorithms,envs,core,api,utils}/` lives side-by-side with the legacy `src/{communication,env,rl_algorithms,config}/` until the new pipeline is verified end-to-end with one algorithm (DQN). Remaining algorithms are then migrated one at a time. Legacy code and obsolete config/tests are deleted in bulk in the final phase.

**Tech Stack:** Python 3.10, PyTorch ≥2.0, MiniGrid ≥2.3, Gymnasium ≥0.29, FastAPI ≥0.110, Pydantic ≥2, pytest, PyYAML.

---

## Phase A — Foundation: project layout, packaging, config skeleton

### Task A1: Add `pyproject.toml` declaring the `src/` layout

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write the file**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "marl-nav"
version = "0.1.0"
description = "Beginner-friendly multi-agent RL navigation research codebase"
requires-python = ">=3.10"
readme = "README.md"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "pandas>=2.0",
    "seaborn>=0.13",
    "pillow>=10.0",
    "pyyaml>=6.0",
    "minigrid>=2.3.0",
    "gymnasium>=0.29",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "httpx>=0.27"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Install in editable mode**

Run: `pip install -e .`
Expected: install succeeds; `pip show marl-nav` lists the package.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml declaring src/ layout"
```

---

### Task A2: Create new directory skeleton and `__init__.py` files

**Files:**
- Create: `src/algorithms/__init__.py`
- Create: `src/algorithms/value_based/__init__.py`
- Create: `src/algorithms/policy_based/__init__.py`
- Create: `src/envs/__init__.py`
- Create: `src/core/__init__.py`
- Create: `src/api/__init__.py`
- Create: `src/api/routes/__init__.py`
- Create: `maps/` (directory)
- Create: `config/config.yml` (placeholder, populated in Task A3)

- [ ] **Step 1: Create empty `__init__.py` in each new package directory**

All `__init__.py` files are empty for now.

- [ ] **Step 2: Verify layout**

Run: `python -c "import algorithms, envs, core, api"`
Expected: no error (note: `src/utils/` already exists).

- [ ] **Step 3: Commit**

```bash
git add src/algorithms/ src/envs/ src/core/ src/api/
git commit -m "scaffold: add empty package skeletons for new src layout"
```

---

### Task A3: Author `config/config.yml`

**Files:**
- Create: `config/config.yml` (overwrite placeholder)

- [ ] **Step 1: Write config content**

```yaml
seed: 42

env:
  map_file: "default"
  observation_mode: "partial"   # partial | full
  partial_view_size: 7
  reward_mode: "independent"    # independent | cooperative
  reward_goal: 10.0
  reward_step: -0.01
  reward_collision: -1.0
  reward_team_bonus: 5.0

algorithm:
  name: "dqn"
  lr: 1.0e-4
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

  # on-policy
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

- [ ] **Step 2: Commit**

```bash
git add config/config.yml
git commit -m "config: add single config.yml replacing multi-file base config"
```

---

### Task A4: Author `maps/default.yml`

**Files:**
- Create: `maps/default.yml`

- [ ] **Step 1: Write map definition**

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

- [ ] **Step 2: Commit**

```bash
git add maps/default.yml
git commit -m "config: add default indoor map (3 rooms, 2 doors, 2 agents)"
```

---

## Phase B — Utils: config loader, logger, paths

### Task B1: Implement `src/utils/paths.py`

**Files:**
- Create: `src/utils/paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
from pathlib import Path
from utils.paths import project_root, config_path, map_path

def test_project_root_contains_pyproject():
    assert (project_root() / "pyproject.toml").exists()

def test_config_path_default():
    assert config_path().name == "config.yml"

def test_map_path_by_name():
    p = map_path("default")
    assert p.name == "default.yml"
    assert "maps" in str(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.paths'`.

- [ ] **Step 3: Implement**

```python
# src/utils/paths.py
"""路径工具：基于 pyproject.toml 定位项目根目录。"""
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def project_root() -> Path:
    """向上查找包含 pyproject.toml 的目录作为项目根。"""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Cannot locate project root (no pyproject.toml found)")


def config_path(name: str = "config") -> Path:
    return project_root() / "config" / f"{name}.yml"


def map_path(name: str) -> Path:
    return project_root() / "maps" / f"{name}.yml"


def experiments_dir() -> Path:
    d = project_root() / "experiments"
    d.mkdir(exist_ok=True)
    return d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paths.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/paths.py tests/test_paths.py
git commit -m "feat(utils): add paths helper with project-root discovery"
```

---

### Task B2: Implement `src/utils/config.py`

**Files:**
- Create: `src/utils/config.py`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_loader.py
from utils.config import load_config, merge_overrides


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "env" in cfg
    assert "algorithm" in cfg


def test_merge_overrides_nested():
    base = {"algorithm": {"name": "dqn", "lr": 1e-4}}
    overrides = {"algorithm": {"lr": 5e-5, "epsilon": 0.5}}
    result = merge_overrides(base, overrides)
    assert result["algorithm"]["name"] == "dqn"
    assert result["algorithm"]["lr"] == 5e-5
    assert result["algorithm"]["epsilon"] == 0.5


def test_merge_overrides_does_not_mutate_input():
    base = {"a": {"b": 1}}
    merge_overrides(base, {"a": {"b": 2}})
    assert base["a"]["b"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/utils/config.py
"""配置加载器：读取单一 config.yml + 支持 CLI/API 覆盖。"""
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import config_path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置。默认读取 config/config.yml。"""
    p = Path(path) if path else config_path()
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_overrides(base: dict, overrides: dict) -> dict:
    """递归合并 overrides 到 base 的副本，返回新 dict。"""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_overrides(result[key], value)
        else:
            result[key] = value
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_loader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/config.py tests/test_config_loader.py
git commit -m "feat(utils): add single-file YAML config loader with override merge"
```

---

### Task B3: Implement `src/utils/logger.py`

**Files:**
- Create: `src/utils/logger.py`

- [ ] **Step 1: Implement**

```python
# src/utils/logger.py
"""统一日志工具。输出到 stdout + 可选文件。"""
import logging
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
```

- [ ] **Step 2: Smoke test in shell**

Run: `python -c "from utils.logger import get_logger; get_logger('test').info('hello')"`
Expected: prints `[INFO] test: hello`.

- [ ] **Step 3: Commit**

```bash
git add src/utils/logger.py
git commit -m "feat(utils): add unified logger with optional file output"
```

---

## Phase C — Environment: MiniGrid wrapper

### Task C1: Verify MiniGrid install

- [ ] **Step 1: Install**

Run: `pip install minigrid gymnasium`
Expected: success.

- [ ] **Step 2: Smoke check**

Run: `python -c "import minigrid; import gymnasium as gym; env = gym.make('MiniGrid-Empty-5x5-v0'); print(env.observation_space)"`
Expected: prints a `Dict` observation space.

- [ ] **Step 3: No commit** (pyproject.toml already lists these).

---

### Task C2: Implement `src/envs/map_builder.py`

**Files:**
- Create: `src/envs/map_builder.py`
- Test: `tests/test_map_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_map_builder.py
import numpy as np
from envs.map_builder import load_map_spec, build_grid_array


def test_load_map_spec_returns_dict():
    spec = load_map_spec("default")
    assert spec["size"] == [15, 15]
    assert spec["num_agents"] == 2
    assert len(spec["rooms"]) == 3
    assert len(spec["agents_start"]) == 2


def test_build_grid_array_shape():
    spec = load_map_spec("default")
    grid = build_grid_array(spec)
    assert grid.shape == (15, 15)


def test_build_grid_array_has_walls_around_rooms():
    spec = load_map_spec("default")
    grid = build_grid_array(spec)
    # 房间分隔墙在 row=7 处（room_1 高度 7，room_3 从 row=8 开始）
    assert (grid[7, :] == 1).sum() > 0  # 至少有部分墙体


def test_build_grid_array_has_doors_as_open():
    spec = load_map_spec("default")
    grid = build_grid_array(spec)
    for door in spec["doors"]:
        r, c = door["position"]
        # 门位置应该是可通行的（值 != 1=墙）
        assert grid[r, c] != 1, f"Door at {(r, c)} is blocked"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_map_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/envs/map_builder.py
"""地图加载与构建：从 yml 读取房间/门/墙定义，生成 numpy 网格。

网格编码：
    0 = 空地（可通行）
    1 = 墙
    2 = 门（可通行，仅用于可视化区分）
"""
from typing import Any

import numpy as np
import yaml

from utils.paths import map_path

EMPTY = 0
WALL = 1
DOOR = 2


def load_map_spec(name: str) -> dict[str, Any]:
    """加载地图 yml，返回 `map` 段下的 dict。"""
    with open(map_path(name), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["map"]


def build_grid_array(spec: dict) -> np.ndarray:
    """根据 spec 构建网格：先填墙，再按房间挖空，最后开门。"""
    rows, cols = spec["size"]
    grid = np.full((rows, cols), WALL, dtype=np.int8)

    # 房间内部置空
    for room in spec["rooms"]:
        r0, c0 = room["top_left"]
        h, w = room["size"]
        r1 = min(r0 + h, rows)
        c1 = min(c0 + w, cols)
        grid[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = EMPTY

    # 外边界一圈墙保留；房间边界已经是墙
    # 在门的位置开洞
    for door in spec["doors"]:
        r, c = door["position"]
        grid[r, c] = DOOR
    return grid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_map_builder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/envs/map_builder.py tests/test_map_builder.py
git commit -m "feat(envs): add map_builder loading yml and building numpy grid"
```

---

### Task C3: Implement `src/envs/indoor_env.py` (multi-agent MiniGrid wrapper)

**Files:**
- Create: `src/envs/indoor_env.py`
- Modify: `src/envs/__init__.py` (export `IndoorEnv`)
- Test: `tests/test_indoor_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_indoor_env.py
import numpy as np
from envs.indoor_env import IndoorEnv


def make_cfg(observation_mode="full", reward_mode="independent"):
    return {
        "env": {
            "map_file": "default",
            "observation_mode": observation_mode,
            "partial_view_size": 7,
            "reward_mode": reward_mode,
            "reward_goal": 10.0,
            "reward_step": -0.01,
            "reward_collision": -1.0,
            "reward_team_bonus": 5.0,
        },
        "seed": 42,
    }


def test_reset_returns_dict_per_agent():
    env = IndoorEnv(make_cfg())
    obs = env.reset()
    assert set(obs.keys()) == {0, 1}
    assert obs[0].shape == obs[1].shape


def test_step_returns_four_dicts():
    env = IndoorEnv(make_cfg())
    env.reset()
    actions = {0: 0, 1: 1}
    obs, rewards, dones, infos = env.step(actions)
    for d in (obs, rewards, dones, infos):
        assert set(d.keys()) == {0, 1}


def test_action_space_has_four_directions():
    env = IndoorEnv(make_cfg())
    assert env.action_space.n == 4


def test_partial_observation_size():
    env = IndoorEnv(make_cfg(observation_mode="partial"))
    obs = env.reset()
    # 7x7 视野 flatten 后 = 49
    assert obs[0].shape[-1] == 49


def test_full_observation_size():
    env = IndoorEnv(make_cfg(observation_mode="full"))
    obs = env.reset()
    # 15x15 = 225
    assert obs[0].shape[-1] == 225


def test_reaching_goal_gives_goal_reward():
    env = IndoorEnv(make_cfg())
    env.reset()
    env.agent_positions[0] = list(env.spec["goals"][0])
    env.agent_positions[0][1] -= 1  # 右边一步到达 goal
    _, rewards, dones, _ = env.step({0: 0, 1: 3})  # 假设动作 0 = 向右
    # 至少一个智能体到达目标
    assert any(r >= 10.0 - 0.1 for r in rewards.values()) or dones[0]


def test_cooperative_team_bonus():
    cfg = make_cfg(reward_mode="cooperative")
    env = IndoorEnv(cfg)
    env.reset()
    # 强行把两个智能体放到目标位置
    env.agent_positions[0] = list(env.spec["goals"][0])
    env.agent_positions[1] = list(env.spec["goals"][1])
    env._goal_reached = {0: True, 1: True}
    # 触发 team bonus 检查
    bonus = env._compute_team_bonus()
    assert bonus == cfg["env"]["reward_team_bonus"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_indoor_env.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/envs/indoor_env.py
"""多智能体室内导航环境（基于自构建的 numpy 网格 + Gymnasium 接口）。

注：MiniGrid 原生为单智能体，为了支持多智能体协同，本类直接基于 map_builder
生成的 numpy 网格实现 reset/step，沿用 MiniGrid 的渲染思想但简化逻辑。
"""
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.map_builder import EMPTY, WALL, DOOR, build_grid_array, load_map_spec

# 动作编码
ACTION_RIGHT = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_UP = 3
ACTION_DELTAS = {
    ACTION_RIGHT: (0, 1),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_UP: (-1, 0),
}


class IndoorEnv(gym.Env):
    """多智能体室内导航环境。

    obs:    dict[agent_id, np.ndarray]
    action: dict[agent_id, int]   动作为 0..3 表示四个方向
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        env_cfg = cfg["env"]
        self.spec = load_map_spec(env_cfg["map_file"])
        self.grid = build_grid_array(self.spec)
        self.rows, self.cols = self.grid.shape
        self.num_agents = self.spec["num_agents"]

        self.observation_mode = env_cfg["observation_mode"]
        self.view_size = env_cfg["partial_view_size"]
        self.reward_mode = env_cfg["reward_mode"]
        self.r_goal = env_cfg["reward_goal"]
        self.r_step = env_cfg["reward_step"]
        self.r_collision = env_cfg["reward_collision"]
        self.r_team = env_cfg["reward_team_bonus"]

        self.action_space = spaces.Discrete(4)
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        seed = cfg.get("seed", None)
        self._rng = np.random.default_rng(seed)
        self.agent_positions: list[list[int]] = []
        self._goal_reached: dict[int, bool] = {}

    def _obs_dim(self) -> int:
        if self.observation_mode == "partial":
            return self.view_size * self.view_size
        return self.rows * self.cols

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agent_positions = [list(p) for p in self.spec["agents_start"]]
        self._goal_reached = {i: False for i in range(self.num_agents)}
        return self._get_obs()

    def step(self, actions: dict[int, int]):
        rewards = {i: 0.0 for i in range(self.num_agents)}
        dones = {i: self._goal_reached[i] for i in range(self.num_agents)}
        infos = {i: {} for i in range(self.num_agents)}

        for agent_id, action in actions.items():
            if self._goal_reached[agent_id]:
                continue
            dr, dc = ACTION_DELTAS[action]
            r, c = self.agent_positions[agent_id]
            nr, nc = r + dr, c + dc
            if not self._in_bounds(nr, nc) or self.grid[nr, nc] == WALL:
                rewards[agent_id] += self.r_collision
                infos[agent_id]["collision"] = True
            else:
                self.agent_positions[agent_id] = [nr, nc]
            rewards[agent_id] += self.r_step

            goal = self.spec["goals"][agent_id]
            if self.agent_positions[agent_id] == list(goal):
                rewards[agent_id] += self.r_goal
                self._goal_reached[agent_id] = True
                dones[agent_id] = True

        if self.reward_mode == "cooperative":
            bonus = self._compute_team_bonus()
            if bonus > 0:
                for i in rewards:
                    rewards[i] += bonus

        return self._get_obs(), rewards, dones, infos

    def _compute_team_bonus(self) -> float:
        if all(self._goal_reached.values()):
            return self.r_team
        return 0.0

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _get_obs(self) -> dict[int, np.ndarray]:
        obs = {}
        for i in range(self.num_agents):
            if self.observation_mode == "partial":
                obs[i] = self._partial_view(i).flatten().astype(np.float32)
            else:
                obs[i] = self._full_view(i).flatten().astype(np.float32)
        return obs

    def _partial_view(self, agent_id: int) -> np.ndarray:
        r, c = self.agent_positions[agent_id]
        half = self.view_size // 2
        view = np.ones((self.view_size, self.view_size), dtype=np.int8) * WALL
        for i in range(self.view_size):
            for j in range(self.view_size):
                gr, gc = r - half + i, c - half + j
                if self._in_bounds(gr, gc):
                    view[i, j] = self.grid[gr, gc]
        # 将自身位置标记
        view[half, half] = 3
        return view

    def _full_view(self, agent_id: int) -> np.ndarray:
        view = self.grid.copy()
        for i, pos in enumerate(self.agent_positions):
            view[pos[0], pos[1]] = 3 if i == agent_id else 4
        return view

    def render(self) -> np.ndarray:
        """返回 RGB 数组用于可视化（简单灰度映射）。"""
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        rgb[self.grid == EMPTY] = (240, 240, 240)
        rgb[self.grid == WALL] = (60, 60, 60)
        rgb[self.grid == DOOR] = (180, 140, 40)
        for i, (r, c) in enumerate(self.agent_positions):
            rgb[r, c] = (50, 100, 220) if i == 0 else (220, 50, 50)
        for i, (r, c) in enumerate(self.spec["goals"]):
            if self.grid[r, c] in (EMPTY, DOOR):
                rgb[r, c] = (50, 200, 50)
        return rgb
```

- [ ] **Step 4: Export from package**

```python
# src/envs/__init__.py
from envs.indoor_env import IndoorEnv

__all__ = ["IndoorEnv"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_indoor_env.py -v`
Expected: PASS (some tests may need minor adjustment if action codes differ).

- [ ] **Step 6: Commit**

```bash
git add src/envs/indoor_env.py src/envs/__init__.py tests/test_indoor_env.py
git commit -m "feat(envs): add multi-agent IndoorEnv with configurable obs and reward modes"
```

---

## Phase D — Vertical slice: DQN end-to-end

### Task D1: Implement `src/algorithms/base.py`

**Files:**
- Create: `src/algorithms/base.py`

- [ ] **Step 1: Implement**

```python
# src/algorithms/base.py
"""所有算法的抽象基类。统一接口便于训练/测试循环复用。"""
from abc import ABC, abstractmethod
from typing import Any


class BaseAlgorithm(ABC):
    """RL 算法基类。

    所有算法需实现 take_action / update / save / load 四个方法。
    """

    @abstractmethod
    def take_action(
        self, states: dict[int, Any], explore: bool = True
    ) -> dict[int, int]:
        """根据当前观测返回每个智能体的动作。"""

    @abstractmethod
    def update(self, *args, **kwargs) -> dict[str, float]:
        """执行一次参数更新，返回 loss 等训练指标的字典。"""

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型权重到指定路径。"""

    @abstractmethod
    def load(self, path: str) -> None:
        """从指定路径加载模型权重。"""

    @property
    def is_on_policy(self) -> bool:
        """on/off-policy 标识，trainer 用此选择训练流程。"""
        return False
```

- [ ] **Step 2: Commit**

```bash
git add src/algorithms/base.py
git commit -m "feat(algorithms): add BaseAlgorithm abstract interface"
```

---

### Task D2: Implement `src/core/replay.py`

**Files:**
- Create: `src/core/replay.py`
- Test: `tests/test_replay.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replay.py
import numpy as np
from core.replay import ReplayBuffer, JointReplayBuffer


def test_replay_buffer_basic():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.push(np.zeros(5), 0, 1.0, np.zeros(5), False)
    assert len(buf) == 10
    batch = buf.sample(4)
    assert len(batch) == 5  # s, a, r, s', done
    assert batch[0].shape == (4, 5)


def test_replay_buffer_capacity():
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(np.array([i]), 0, 0.0, np.array([i + 1]), False)
    assert len(buf) == 5


def test_joint_replay_buffer_basic():
    buf = JointReplayBuffer(capacity=100, num_agents=2, state_dim=5)
    for _ in range(10):
        joint_s = {0: np.zeros(5), 1: np.zeros(5)}
        joint_a = {0: 0, 1: 1}
        joint_r = {0: 1.0, 1: 0.5}
        joint_s_next = {0: np.zeros(5), 1: np.zeros(5)}
        joint_done = {0: False, 1: False}
        buf.push(joint_s, joint_a, joint_r, joint_s_next, joint_done)
    assert len(buf) == 10
    batch = buf.sample(4)
    states, actions, rewards, next_states, dones = batch
    assert states.shape == (4, 2, 5)
    assert actions.shape == (4, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/core/replay.py
"""经验回放缓冲区。

ReplayBuffer：标准单智能体（DQN/MADQN 中按 agent 各持一份）
JointReplayBuffer：QMIX/VDN 用，存联合 (s, a, r, s', done)
"""
import random
from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class JointReplayBuffer:
    """存储所有智能体联合的 (s, a, r, s', done)，shape = (B, N, *)."""

    def __init__(self, capacity: int, num_agents: int, state_dim: int):
        self.capacity = capacity
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        states: dict[int, np.ndarray],
        actions: dict[int, int],
        rewards: dict[int, float],
        next_states: dict[int, np.ndarray],
        dones: dict[int, bool],
    ) -> None:
        s = np.stack([states[i] for i in range(self.num_agents)])
        a = np.array([actions[i] for i in range(self.num_agents)], dtype=np.int64)
        r = np.array([rewards[i] for i in range(self.num_agents)], dtype=np.float32)
        s_next = np.stack([next_states[i] for i in range(self.num_agents)])
        d = np.array([dones[i] for i in range(self.num_agents)], dtype=np.float32)
        self.buffer.append((s, a, r, s_next, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (
            np.stack(s),
            np.stack(a),
            np.stack(r),
            np.stack(s_next),
            np.stack(d),
        )

    def __len__(self) -> int:
        return len(self.buffer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/replay.py tests/test_replay.py
git commit -m "feat(core): add ReplayBuffer and JointReplayBuffer"
```

---

### Task D3: Implement DQN (`src/algorithms/value_based/dqn/`)

**Files:**
- Create: `src/algorithms/value_based/dqn/__init__.py`
- Create: `src/algorithms/value_based/dqn/qnet.py`
- Create: `src/algorithms/value_based/dqn/algo.py`

- [ ] **Step 1: Implement Q-network**

```python
# src/algorithms/value_based/dqn/qnet.py
"""DQN 的 Q 网络：两层 MLP。"""
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 2: Implement DQN algorithm**

```python
# src/algorithms/value_based/dqn/algo.py
"""标准 DQN：epsilon-greedy 探索 + 目标网络 + 经验回放。

注：在多智能体环境中，本类只控制 agent 0，其他智能体随机动作。
入门理解 Q-learning 流程的首选算法。
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.value_based.dqn.qnet import QNet


class DQN(BaseAlgorithm):
    def __init__(self, env, cfg: dict):
        algo_cfg = cfg["algorithm"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = algo_cfg["gamma"]
        self.lr = algo_cfg["lr"]
        self.epsilon = algo_cfg["epsilon"]
        self.epsilon_min = algo_cfg["epsilon_min"]
        self.epsilon_decay = algo_cfg["epsilon_decay"]
        self.update_freq = algo_cfg["update_freq"]
        self._update_count = 0

        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.num_agents = env.num_agents
        self.controlled_agent = 0  # 仅训练 agent 0

        self.q = QNet(self.state_dim, algo_cfg["hidden_dim"], self.n_actions).to(
            self.device
        )
        self.q_target = deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

    def take_action(self, states: dict[int, np.ndarray], explore: bool = True):
        actions = {}
        for agent_id in range(self.num_agents):
            if agent_id == self.controlled_agent:
                if explore and np.random.rand() < self.epsilon:
                    actions[agent_id] = np.random.randint(self.n_actions)
                else:
                    s = torch.from_numpy(states[agent_id]).float().unsqueeze(0).to(
                        self.device
                    )
                    with torch.no_grad():
                        q_values = self.q(s)
                    actions[agent_id] = int(q_values.argmax(dim=1).item())
            else:
                actions[agent_id] = np.random.randint(self.n_actions)
        return actions

    def update(self, batch) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        q = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(next_states).max(dim=1)[0]
            target = rewards + self.gamma * q_next * (1 - dones)
        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target.load_state_dict(self.q.state_dict())
```

- [ ] **Step 3: Add `__init__.py`**

```python
# src/algorithms/value_based/dqn/__init__.py
from algorithms.value_based.dqn.algo import DQN

__all__ = ["DQN"]
```

- [ ] **Step 4: Commit**

```bash
git add src/algorithms/value_based/dqn/
git commit -m "feat(algorithms): add DQN (single-controlled-agent baseline)"
```

---

### Task D4: Implement algorithm registry

**Files:**
- Modify: `src/algorithms/__init__.py`

- [ ] **Step 1: Implement registry**

```python
# src/algorithms/__init__.py
"""算法注册表。新增算法只需在此添加一行，无需修改 trainer。"""
from algorithms.base import BaseAlgorithm
from algorithms.value_based.dqn import DQN

ALGORITHM_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    "dqn": DQN,
}


def build_algorithm(name: str, env, cfg: dict) -> BaseAlgorithm:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {name!r}. Available: {sorted(ALGORITHM_REGISTRY)}"
        )
    return ALGORITHM_REGISTRY[name](env, cfg)


__all__ = ["BaseAlgorithm", "ALGORITHM_REGISTRY", "build_algorithm"]
```

- [ ] **Step 2: Smoke test**

Run: `python -c "from algorithms import build_algorithm; print(build_algorithm.__doc__ or 'OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/algorithms/__init__.py
git commit -m "feat(algorithms): add ALGORITHM_REGISTRY and build_algorithm factory"
```

---

### Task D5: Implement `src/core/trainer.py`

**Files:**
- Create: `src/core/trainer.py`

- [ ] **Step 1: Implement**

```python
# src/core/trainer.py
"""统一训练循环。

支持 on/off-policy 两种范式，通过 algo.is_on_policy 切换。
当前阶段先实现 off-policy 路径，on-policy 路径在 PPO 任务中补充。
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from algorithms.base import BaseAlgorithm
from core.replay import ReplayBuffer, JointReplayBuffer


@dataclass
class TrainResult:
    run_id: str
    model_path: str
    history: list[dict] = field(default_factory=list)


def _run_episode_off_policy(
    algo: BaseAlgorithm,
    env,
    buffer,
    max_steps: int,
    train_interval: int,
    batch_size: int,
    learn: bool,
) -> dict:
    states = env.reset()
    total_reward = 0.0
    step_count = 0
    losses: list[float] = []

    for step in range(max_steps):
        actions = algo.take_action(states, explore=True)
        next_states, rewards, dones, _ = env.step(actions)
        step_count += 1
        ep_r = sum(rewards.values())
        total_reward += ep_r

        # 仅推入受控 agent 的经验
        ca = algo.controlled_agent
        buffer.push(
            states[ca], actions[ca], rewards[ca], next_states[ca], dones[ca]
        )

        states = next_states
        if learn and len(buffer) >= batch_size and step % train_interval == 0:
            batch = buffer.sample(batch_size)
            info = algo.update(batch)
            losses.append(info.get("loss", 0.0))
        if all(dones.values()):
            break

    return {
        "reward": total_reward,
        "steps": step_count,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }


def train(
    algo: BaseAlgorithm,
    env,
    cfg: dict,
    run_dir: Path,
) -> TrainResult:
    algo_cfg = cfg["algorithm"]
    num_episodes = algo_cfg["num_episodes"]
    episode_length = algo_cfg["episode_length"]
    batch_size = algo_cfg["batch_size"]
    train_interval = algo_cfg["train_interval"]

    buffer = ReplayBuffer(algo_cfg["replay_buffer_size"])
    history: list[dict] = []

    for ep in range(num_episodes):
        t0 = time.time()
        info = _run_episode_off_policy(
            algo, env, buffer, episode_length, train_interval, batch_size, learn=True
        )
        info["episode"] = ep
        info["wall_time"] = time.time() - t0
        history.append(info)

        if ep % cfg["logging"]["save_interval"] == 0 or ep == num_episodes - 1:
            ckpt = run_dir / "model.pth"
            algo.save(str(ckpt))

    run_id = run_dir.name
    return TrainResult(
        run_id=run_id, model_path=str(run_dir / "model.pth"), history=history
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/core/trainer.py
git commit -m "feat(core): add off-policy train() loop"
```

---

### Task D6: Implement `src/core/plot.py`

**Files:**
- Create: `src/core/plot.py`

- [ ] **Step 1: Implement**

```python
# src/core/plot.py
"""训练曲线绘图：episode_reward + steps_to_goal。"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["axes.unicode_minus"] = False


def plot_training(history: list[dict], out_dir: Path) -> None:
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = [h["episode"] for h in history]
    rewards = [h["reward"] for h in history]
    steps = [h["steps"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps, rewards, label="episode reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("Training reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps, steps, label="steps")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")
    ax.set_title("Steps per episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "steps.png", dpi=120)
    plt.close(fig)
```

- [ ] **Step 2: Commit**

```bash
git add src/core/plot.py
git commit -m "feat(core): add training curve plotting"
```

---

### Task D7: Implement `src/core/tester.py`

**Files:**
- Create: `src/core/tester.py`

- [ ] **Step 1: Implement**

```python
# src/core/tester.py
"""测试单次 episode 并生成 gif/png 可视化。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from algorithms.base import BaseAlgorithm


@dataclass
class TestResult:
    success: bool
    steps: int
    total_reward: float
    gif_path: str
    png_path: str


def test(
    algo: BaseAlgorithm, env, cfg: dict, out_dir: Path, max_steps: int = 500
) -> TestResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    states = env.reset()
    frames: list[np.ndarray] = [env.render()]
    total_reward = 0.0
    success = False

    for step in range(max_steps):
        actions = algo.take_action(states, explore=False)
        states, rewards, dones, _ = env.step(actions)
        frames.append(env.render())
        total_reward += sum(rewards.values())
        if all(dones.values()):
            success = True
            break

    gif_path = out_dir / "nav.gif"
    png_path = out_dir / "nav_final.png"
    pil_frames = [Image.fromarray(f).resize((300, 300), Image.NEAREST) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=200,
        loop=0,
    )
    pil_frames[-1].save(png_path)
    return TestResult(
        success=success,
        steps=step + 1,
        total_reward=total_reward,
        gif_path=str(gif_path),
        png_path=str(png_path),
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/core/tester.py
git commit -m "feat(core): add test() with gif/png visualization"
```

---

### Task D8: Implement CLI `main.py`

**Files:**
- Create: `main.py` (overwrite legacy if exists, but first inspect)

- [ ] **Step 1: Inspect existing `main.py`**

Run: `cat main.py | head -30` to confirm what we're replacing.
We'll replace it entirely; legacy code stays only in `src/rl_algorithms/`.

- [ ] **Step 2: Implement**

```python
# main.py
"""CLI 入口：训练 / 测试 RL 算法在室内多智能体环境中。"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from algorithms import build_algorithm
from core.plot import plot_training
from core.tester import test
from core.trainer import train
from envs import IndoorEnv
from utils.config import load_config, merge_overrides
from utils.logger import get_logger
from utils.paths import experiments_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, help="algorithm name from registry")
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--map", default=None, help="map file name (without .yml)")
    p.add_argument("--model_path", default=None, help="checkpoint path for test mode")
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tag", default="", help="appended to run id")
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_run_dir(algo: str, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    d = experiments_dir() / f"{stamp}_{algo}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    args = _parse_args()
    cfg = load_config()

    overrides: dict = {"algorithm": {"name": args.algo}}
    if args.map:
        overrides["env"] = {"map_file": args.map}
    if args.num_episodes is not None:
        overrides["algorithm"]["num_episodes"] = args.num_episodes
    if args.lr is not None:
        overrides["algorithm"]["lr"] = args.lr
    cfg = merge_overrides(cfg, overrides)

    _seed_everything(cfg["seed"])
    run_dir = _make_run_dir(args.algo, args.tag)
    logger = get_logger("main", log_file=run_dir / "run.log")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Algorithm: {args.algo}, mode: {args.mode}")

    env = IndoorEnv(cfg)
    algo = build_algorithm(args.algo, env, cfg)

    if args.mode == "train":
        result = train(algo, env, cfg, run_dir)
        plot_training(result.history, run_dir / "figs")
        logger.info(f"Training complete. Model: {result.model_path}")
    else:
        if args.model_path:
            algo.load(args.model_path)
        result = test(algo, env, cfg, run_dir / "test", max_steps=args.max_steps)
        logger.info(
            f"Test: success={result.success} steps={result.steps} "
            f"reward={result.total_reward:.2f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test (short train)**

Run: `python main.py --algo dqn --mode train --num_episodes 5`
Expected: completes without error; creates `experiments/<stamp>_dqn/model.pth` and `figs/reward.png`.

- [ ] **Step 4: Smoke test (test mode)**

Run: `python main.py --algo dqn --mode test --model_path experiments/<stamp>_dqn/model.pth`
Expected: creates `nav.gif`.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: replace legacy main.py with new CLI for refactored pipeline"
```

---

## Phase E — Migrate remaining algorithms

For each algorithm, the recipe is:
1. **Copy** old `algo.py` + `qnet.py` (and `mixer.py` for QMIX) from `src/rl_algorithms/<name>/` to `src/algorithms/{value_based|policy_based}/<name>/`.
2. **Adapt imports** to the new package paths.
3. **Adapt to `BaseAlgorithm`** interface: ensure `take_action(states: dict)` returns `dict[int, int]`, `update()` accepts a batch tuple matching `ReplayBuffer.sample()` (or `JointReplayBuffer.sample()` for CTDE).
4. **Strip all comm-related code**: no `ber_*`, no `power_indices`, no `target_idx`, no per-step BER tracking.
5. **Register** in `ALGORITHM_REGISTRY`.
6. **Smoke test** with `python main.py --algo <name> --mode train --num_episodes 5`.
7. **Commit** as `feat(algorithms): migrate <name> to new package`.

### Task E1: Migrate MADQN

**Files:**
- Create: `src/algorithms/value_based/madqn/__init__.py`
- Create: `src/algorithms/value_based/madqn/qnet.py`
- Create: `src/algorithms/value_based/madqn/algo.py`
- Modify: `src/algorithms/__init__.py` (add to registry)
- Modify: `src/core/trainer.py` (extend to MADQN — per-agent replay buffers)

- [ ] **Step 1: Read source**

Read `src/rl_algorithms/madqn/algo.py` and `qnet.py` to understand current structure.

- [ ] **Step 2: Copy and adapt qnet**

```python
# src/algorithms/value_based/madqn/qnet.py
"""MADQN 的 Q 网络与 DQN 相同（每个 agent 独立持有一份）。"""
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 3: Implement MADQN algo**

```python
# src/algorithms/value_based/madqn/algo.py
"""独立多智能体 DQN：每个 agent 一个 Q 网络与目标网络。"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.value_based.madqn.qnet import QNet


class MADQN(BaseAlgorithm):
    def __init__(self, env, cfg: dict):
        algo_cfg = cfg["algorithm"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = algo_cfg["gamma"]
        self.lr = algo_cfg["lr"]
        self.epsilon = algo_cfg["epsilon"]
        self.epsilon_min = algo_cfg["epsilon_min"]
        self.epsilon_decay = algo_cfg["epsilon_decay"]
        self.update_freq = algo_cfg["update_freq"]
        self._update_count = 0

        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.num_agents = env.num_agents

        self.qs = [
            QNet(self.state_dim, algo_cfg["hidden_dim"], self.n_actions).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.q_targets = [deepcopy(q) for q in self.qs]
        for q_t in self.q_targets:
            for p in q_t.parameters():
                p.requires_grad = False
        self.optimizers = [torch.optim.Adam(q.parameters(), lr=self.lr) for q in self.qs]

    def take_action(self, states, explore=True):
        actions = {}
        for i in range(self.num_agents):
            if explore and np.random.rand() < self.epsilon:
                actions[i] = np.random.randint(self.n_actions)
            else:
                s = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    actions[i] = int(self.qs[i](s).argmax(dim=1).item())
        return actions

    def update(self, batches: dict[int, tuple]) -> dict[str, float]:
        """batches: {agent_id: (s, a, r, s', done)}"""
        losses = []
        for i in range(self.num_agents):
            s, a, r, s_next, d = (torch.from_numpy(x).to(self.device) for x in batches[i])
            s = s.float()
            s_next = s_next.float()
            a = a.long()
            r = r.float()
            d = d.float()

            q = self.qs[i](s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = self.q_targets[i](s_next).max(dim=1)[0]
                target = r + self.gamma * q_next * (1 - d)
            loss = F.mse_loss(q, target)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()
            losses.append(float(loss.item()))

        self._update_count += 1
        if self._update_count % self.update_freq == 0:
            for q, q_t in zip(self.qs, self.q_targets):
                q_t.load_state_dict(q.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"loss": float(np.mean(losses)), "epsilon": float(self.epsilon)}

    def save(self, path: str) -> None:
        torch.save([q.state_dict() for q in self.qs], path)

    def load(self, path: str) -> None:
        states = torch.load(path, map_location=self.device)
        for q, sd in zip(self.qs, states):
            q.load_state_dict(sd)
        for q, q_t in zip(self.qs, self.q_targets):
            q_t.load_state_dict(q.state_dict())
```

- [ ] **Step 4: Extend trainer to handle per-agent buffers**

Modify `src/core/trainer.py` — add an `if isinstance(algo, MADQN)` branch is bad form; instead, add a method on the algorithm:

```python
# Add to BaseAlgorithm in src/algorithms/base.py:
def required_buffer(self) -> str:
    """Return 'single' | 'per_agent' | 'joint' | 'none'."""
    return "single"
```

Then in DQN return `"single"`, in MADQN return `"per_agent"`, in QMIX/VDN return `"joint"`, in PPO/MAPPO return `"none"`.

Update `src/core/trainer.py`:

```python
def _make_buffer(algo, env, algo_cfg):
    kind = algo.required_buffer()
    cap = algo_cfg["replay_buffer_size"]
    if kind == "single":
        return ReplayBuffer(cap)
    if kind == "per_agent":
        return {i: ReplayBuffer(cap) for i in range(env.num_agents)}
    if kind == "joint":
        return JointReplayBuffer(cap, env.num_agents, env.observation_space.shape[0])
    return None


def _push(buffer, kind, states, actions, rewards, next_states, dones, controlled):
    if kind == "single":
        a = controlled
        buffer.push(states[a], actions[a], rewards[a], next_states[a], dones[a])
    elif kind == "per_agent":
        for i in buffer:
            buffer[i].push(states[i], actions[i], rewards[i], next_states[i], dones[i])
    elif kind == "joint":
        buffer.push(states, actions, rewards, next_states, dones)


def _sample(buffer, kind, batch_size):
    if kind == "single":
        return buffer.sample(batch_size)
    if kind == "per_agent":
        return {i: buffer[i].sample(batch_size) for i in buffer}
    if kind == "joint":
        return buffer.sample(batch_size)


def _ready(buffer, kind, batch_size):
    if kind == "single" or kind == "joint":
        return len(buffer) >= batch_size
    if kind == "per_agent":
        return all(len(b) >= batch_size for b in buffer.values())
    return False
```

Refactor `_run_episode_off_policy` to use these helpers.

- [ ] **Step 5: Register MADQN**

In `src/algorithms/__init__.py`:

```python
from algorithms.value_based.madqn import MADQN
ALGORITHM_REGISTRY["madqn"] = MADQN
```

And add `__init__.py`:

```python
# src/algorithms/value_based/madqn/__init__.py
from algorithms.value_based.madqn.algo import MADQN
__all__ = ["MADQN"]
```

- [ ] **Step 6: Smoke test**

Run: `python main.py --algo madqn --mode train --num_episodes 5`
Expected: completes; creates checkpoint.

- [ ] **Step 7: Commit**

```bash
git add src/algorithms/value_based/madqn/ src/algorithms/__init__.py src/algorithms/base.py src/core/trainer.py
git commit -m "feat(algorithms): migrate MADQN with per-agent replay buffer support"
```

---

### Task E2: Migrate SharedMADQN

Follow the same recipe as Task E1. Key difference: a single shared `QNet` controls all agents, optimizer shared.

- [ ] **Step 1: Read source** `src/rl_algorithms/shared_madqn/`
- [ ] **Step 2: Copy & adapt** to `src/algorithms/value_based/shared_madqn/`
- [ ] **Step 3: Strip comm code** (target_idx, etc.)
- [ ] **Step 4: `required_buffer()` returns `"per_agent"`** (or `"single"` since shared net — your call; per_agent is simpler since training batches still come from individual agent trajectories)
- [ ] **Step 5: Register** in `ALGORITHM_REGISTRY`
- [ ] **Step 6: Smoke test** `python main.py --algo shared_madqn --mode train --num_episodes 5`
- [ ] **Step 7: Commit** `feat(algorithms): migrate SharedMADQN`

---

### Task E3: Migrate VDN

VDN sums per-agent Q-values then computes a single TD loss. Needs `JointReplayBuffer`.

- [ ] **Step 1: Read source** `src/rl_algorithms/vdn/`
- [ ] **Step 2: Copy & adapt** to `src/algorithms/value_based/vdn/`
- [ ] **Step 3: Strip comm code**
- [ ] **Step 4: `required_buffer()` returns `"joint"`**
- [ ] **Step 5: Update `take_action(states: dict) -> dict[int, int]`** and `update(batch)` where batch shape is `(B, N, *)`
- [ ] **Step 6: Register** in `ALGORITHM_REGISTRY`
- [ ] **Step 7: Smoke test** `python main.py --algo vdn --mode train --num_episodes 5`
- [ ] **Step 8: Commit** `feat(algorithms): migrate VDN with joint replay`

---

### Task E4: Migrate QMIX

QMIX = VDN + monotonic mixer network. Has 3 files: `algo.py`, `qnet.py`, `mixer.py`.

- [ ] **Step 1: Read source** `src/rl_algorithms/qmix/`
- [ ] **Step 2: Copy** all three files to `src/algorithms/value_based/qmix/`
- [ ] **Step 3: Adapt imports + strip comm code**
- [ ] **Step 4: `required_buffer()` returns `"joint"`**
- [ ] **Step 5: Update `take_action`/`update` signatures**
- [ ] **Step 6: Register** in `ALGORITHM_REGISTRY`
- [ ] **Step 7: Smoke test** `python main.py --algo qmix --mode train --num_episodes 5`
- [ ] **Step 8: Commit** `feat(algorithms): migrate QMIX (mixer + monotonic constraint)`

---

### Task E5: Migrate PPO (single-agent, on-policy)

PPO needs an on-policy training loop. Add `_run_episode_on_policy` to trainer.

- [ ] **Step 1: Read source** `src/rl_algorithms/ppo/`
- [ ] **Step 2: Copy** `algo.py` + `net.py` to `src/algorithms/policy_based/ppo/`
- [ ] **Step 3: Adapt imports + strip comm**
- [ ] **Step 4: `is_on_policy = True`, `required_buffer() = "none"`**
- [ ] **Step 5: Add on-policy branch in `train()`**

Pseudocode:

```python
def train(algo, env, cfg, run_dir):
    if algo.is_on_policy:
        return _train_on_policy(algo, env, cfg, run_dir)
    return _train_off_policy(algo, env, cfg, run_dir)
```

`_train_on_policy` collects `update_interval` steps of trajectories, calls `algo.update(rollout)` for `ppo_epochs` mini-epochs, repeats.

- [ ] **Step 6: Register** in `ALGORITHM_REGISTRY`
- [ ] **Step 7: Smoke test** `python main.py --algo ppo --mode train --num_episodes 5`
- [ ] **Step 8: Commit** `feat(algorithms): migrate PPO and add on-policy training branch`

---

### Task E6: Migrate MAPPO (multi-agent, on-policy)

- [ ] **Step 1: Read source** `src/rl_algorithms/mappo/`
- [ ] **Step 2: Copy** `algo.py` + `net.py` to `src/algorithms/policy_based/mappo/`
- [ ] **Step 3: Adapt imports + strip comm (including `reward_scale` if comm-specific)**
- [ ] **Step 4: `is_on_policy = True`**
- [ ] **Step 5: Reuse `_train_on_policy`** (it should already handle multi-agent if `take_action` returns dict)
- [ ] **Step 6: Register** in `ALGORITHM_REGISTRY`
- [ ] **Step 7: Smoke test** `python main.py --algo mappo --mode train --num_episodes 5`
- [ ] **Step 8: Commit** `feat(algorithms): migrate MAPPO`

---

## Phase F — FastAPI layer

### Task F1: Implement `src/api/schemas.py`

**Files:**
- Create: `src/api/schemas.py`

- [ ] **Step 1: Implement**

```python
# src/api/schemas.py
"""FastAPI 请求/响应 Pydantic 模型。"""
from typing import Literal

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    algorithm: str = Field(..., description="算法名，必须在 ALGORITHM_REGISTRY 中")
    map_file: str = Field(default="default")
    config_overrides: dict = Field(default_factory=dict)
    tag: str = Field(default="")


class TestRequest(BaseModel):
    algorithm: str
    model_path: str
    map_file: str = "default"
    max_steps: int = 500


class RunIdResponse(BaseModel):
    run_id: str


class StatusResponse(BaseModel):
    run_id: str
    status: Literal["running", "completed", "failed"]
    episode: int
    total_episodes: int
    latest_reward: float
    model_path: str | None = None
    error: str | None = None


class TestResponse(BaseModel):
    success: bool
    steps: int
    total_reward: float
    gif_path: str
    png_path: str
```

- [ ] **Step 2: Commit**

```bash
git add src/api/schemas.py
git commit -m "feat(api): add Pydantic schemas for train/test/status endpoints"
```

---

### Task F2: Implement run registry + background training

**Files:**
- Create: `src/api/runs.py`

- [ ] **Step 1: Implement**

```python
# src/api/runs.py
"""训练任务的运行时注册表与异步后台执行。

设计：内存中维护 {run_id: RunState}，POST /train 时启动后台 asyncio 任务。
重启服务后内存丢失（v1 接受此限制；持久化由网站层决定是否引入）。
"""
import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from algorithms import build_algorithm
from core.trainer import train
from envs import IndoorEnv
from utils.config import load_config, merge_overrides
from utils.paths import experiments_dir


@dataclass
class RunState:
    run_id: str
    status: str = "running"  # running | completed | failed
    episode: int = 0
    total_episodes: int = 0
    latest_reward: float = 0.0
    model_path: Optional[str] = None
    error: Optional[str] = None


_RUNS: dict[str, RunState] = {}


def get_run(run_id: str) -> Optional[RunState]:
    return _RUNS.get(run_id)


def list_runs() -> list[RunState]:
    return list(_RUNS.values())


def _make_run_id(algo: str, tag: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return f"{stamp}_{algo}{suffix}"


async def start_training(
    algorithm: str, map_file: str, overrides: dict, tag: str
) -> str:
    run_id = _make_run_id(algorithm, tag)
    state = RunState(run_id=run_id)
    _RUNS[run_id] = state

    cfg = load_config()
    overrides.setdefault("env", {})["map_file"] = map_file
    overrides.setdefault("algorithm", {})["name"] = algorithm
    cfg = merge_overrides(cfg, overrides)
    state.total_episodes = cfg["algorithm"]["num_episodes"]

    run_dir = experiments_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    asyncio.create_task(_run_train(state, cfg, run_dir))
    return run_id


async def _run_train(state: RunState, cfg: dict, run_dir: Path) -> None:
    loop = asyncio.get_event_loop()
    try:
        # 将阻塞的训练放到线程池，避免阻塞事件循环
        def _do():
            env = IndoorEnv(cfg)
            algo = build_algorithm(cfg["algorithm"]["name"], env, cfg)
            # 包装一个进度回调（通过 closure 更新 state）
            from core.trainer import train as _train
            result = _train(algo, env, cfg, run_dir)
            return result

        result = await loop.run_in_executor(None, _do)
        state.model_path = result.model_path
        if result.history:
            state.episode = result.history[-1]["episode"]
            state.latest_reward = result.history[-1]["reward"]
        state.status = "completed"
    except Exception as e:
        state.error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        state.status = "failed"
```

> **Note for engineer:** Real-time progress updates require trainer to accept a callback; for v1 the status only updates at end-of-training. Documenting this as a known limitation is acceptable.

- [ ] **Step 2: Commit**

```bash
git add src/api/runs.py
git commit -m "feat(api): add in-memory run registry with async background training"
```

---

### Task F3: Implement routes

**Files:**
- Create: `src/api/routes/train.py`
- Create: `src/api/routes/test.py`
- Create: `src/api/routes/status.py`

- [ ] **Step 1: Train route**

```python
# src/api/routes/train.py
from fastapi import APIRouter

from api.runs import start_training
from api.schemas import RunIdResponse, TrainRequest

router = APIRouter()


@router.post("/train", response_model=RunIdResponse)
async def submit_train(req: TrainRequest) -> RunIdResponse:
    run_id = await start_training(
        req.algorithm, req.map_file, req.config_overrides, req.tag
    )
    return RunIdResponse(run_id=run_id)
```

- [ ] **Step 2: Status route**

```python
# src/api/routes/status.py
from fastapi import APIRouter, HTTPException

from api.runs import get_run
from api.schemas import StatusResponse

router = APIRouter()


@router.get("/status/{run_id}", response_model=StatusResponse)
def get_status(run_id: str) -> StatusResponse:
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return StatusResponse(
        run_id=state.run_id,
        status=state.status,
        episode=state.episode,
        total_episodes=state.total_episodes,
        latest_reward=state.latest_reward,
        model_path=state.model_path,
        error=state.error,
    )
```

- [ ] **Step 3: Test route**

```python
# src/api/routes/test.py
from pathlib import Path

from fastapi import APIRouter

from algorithms import build_algorithm
from api.schemas import TestRequest, TestResponse
from core.tester import test
from envs import IndoorEnv
from utils.config import load_config, merge_overrides
from utils.paths import experiments_dir

router = APIRouter()


@router.post("/test", response_model=TestResponse)
def run_test(req: TestRequest) -> TestResponse:
    cfg = load_config()
    cfg = merge_overrides(
        cfg,
        {"env": {"map_file": req.map_file}, "algorithm": {"name": req.algorithm}},
    )
    env = IndoorEnv(cfg)
    algo = build_algorithm(req.algorithm, env, cfg)
    algo.load(req.model_path)
    out_dir = experiments_dir() / f"api_test_{Path(req.model_path).stem}"
    result = test(algo, env, cfg, out_dir, max_steps=req.max_steps)
    return TestResponse(
        success=result.success,
        steps=result.steps,
        total_reward=result.total_reward,
        gif_path=result.gif_path,
        png_path=result.png_path,
    )
```

- [ ] **Step 4: Commit**

```bash
git add src/api/routes/
git commit -m "feat(api): add /train /test /status routes"
```

---

### Task F4: Implement `src/api/main.py`

**Files:**
- Create: `src/api/main.py`

- [ ] **Step 1: Implement**

```python
# src/api/main.py
"""FastAPI app 入口。

启动命令：
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI

from api.routes import status, test, train

app = FastAPI(
    title="MARL-Nav API",
    description="Multi-agent RL navigation algorithm server",
    version="0.1.0",
)
app.include_router(train.router, tags=["train"])
app.include_router(status.router, tags=["status"])
app.include_router(test.router, tags=["test"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 2: Smoke test**

Run: `uvicorn api.main:app --port 8001 &` then `curl http://localhost:8001/health`
Expected: `{"status":"ok"}`. Then kill the process.

- [ ] **Step 3: Commit**

```bash
git add src/api/main.py
git commit -m "feat(api): add FastAPI app entry point with health endpoint"
```

---

### Task F5: API integration test

**Files:**
- Create: `tests/test_api.py`

- [ ] **Step 1: Write test**

```python
# tests/test_api.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_train_returns_run_id():
    payload = {
        "algorithm": "dqn",
        "map_file": "default",
        "config_overrides": {"algorithm": {"num_episodes": 2, "episode_length": 20}},
        "tag": "apitest",
    }
    r = client.post("/train", json=payload)
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert run_id


def test_status_for_unknown_run_id_returns_404():
    r = client.get("/status/nonexistent")
    assert r.status_code == 404
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test(api): add smoke tests for /health /train /status"
```

---

## Phase G — Cleanup of legacy code

### Task G1: Delete the entire communication module

- [ ] **Step 1: Confirm new pipeline is green**

Run: `pytest -x` to ensure all tests pass before deletion.
Run a final smoke train: `python main.py --algo dqn --mode train --num_episodes 3`.

- [ ] **Step 2: Delete files**

```bash
git rm -r src/communication/
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove wireless communication module (no longer needed)"
```

---

### Task G2: Delete legacy `src/env/` and `src/config/`

- [ ] **Step 1: Verify no remaining imports of these packages**

Run: `grep -rn "from env\." src/ tests/ main.py` — must return nothing (only `envs.` allowed).
Run: `grep -rn "from config\." src/ tests/ main.py` — must return nothing.
Run: `grep -rn "from src.config" src/ tests/ main.py` — must return nothing.

If any matches found, fix imports first.

- [ ] **Step 2: Delete**

```bash
git rm -r src/env/ src/config/
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove legacy env and config packages"
```

---

### Task G3: Delete legacy `src/rl_algorithms/`

- [ ] **Step 1: Verify no remaining imports**

Run: `grep -rn "rl_algorithms" src/ tests/ main.py`
Expected: no matches.

- [ ] **Step 2: Delete**

```bash
git rm -r src/rl_algorithms/
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove legacy rl_algorithms package (migrated to algorithms/)"
```

---

### Task G4: Delete obsolete config files

- [ ] **Step 1: Delete**

```bash
git rm -r config/base/ config/dynamic/ 2>/dev/null || true
git rm config/base/*.yml 2>/dev/null || true
```

(Use whichever subset actually exists; check with `git ls-files config/`.)

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove legacy multi-file config (replaced by config/config.yml)"
```

---

### Task G5: Delete obsolete tests and old models

- [ ] **Step 1: Identify obsolete tests**

```bash
git ls-files tests/ | grep -E "(communication|ofdma|test_env\.py|test_env_performance|test_env_visualization|test_config\.py|test_rl_algorithms)"
```

- [ ] **Step 2: Delete matched tests**

```bash
git rm tests/test_communication.py tests/test_ofdma.py tests/test_env.py \
       tests/test_env_performance.py tests/test_env_visualization.py \
       tests/test_config.py tests/test_rl_algorithms.py
```

(Skip any that don't exist.)

- [ ] **Step 3: Delete old model checkpoints**

```bash
git rm -r models/ 2>/dev/null || true
```

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove obsolete tests and pre-refactor model checkpoints"
```

---

### Task G6: Update `requirements.txt`

- [ ] **Step 1: Rewrite to match pyproject.toml**

```
torch>=2.0
numpy>=1.24
matplotlib>=3.7
pandas>=2.0
seaborn>=0.13
pillow>=10.0
pyyaml>=6.0
minigrid>=2.3.0
gymnasium>=0.29
fastapi>=0.110
uvicorn>=0.27
pydantic>=2.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: align requirements.txt with pyproject.toml (drop pygame, add fastapi/minigrid)"
```

---

### Task G7: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README** to describe the new pipeline:

```markdown
# MARL-Nav

Multi-agent RL navigation research codebase. Beginner-friendly pure-algorithm playground
with 7 algorithms (value-based + policy-based) over a configurable indoor grid environment
(rooms, doors, walls). Includes a FastAPI layer for future web deployment.

## Install

```bash
pip install -e .
```

## CLI usage

```bash
python main.py --algo dqn --mode train --num_episodes 500
python main.py --algo madqn --mode test --model_path experiments/<run>/model.pth
```

## API usage

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# POST /train, GET /status/{run_id}, POST /test
```

## Algorithms

- **Value-based:** DQN, MADQN, SharedMADQN, QMIX, VDN
- **Policy-based:** PPO, MAPPO

## Layout

```
src/
├── algorithms/{value_based,policy_based}/<name>/
├── envs/indoor_env.py
├── core/{trainer,tester,replay,plot}.py
├── api/{main,schemas,runs,routes/}.py
└── utils/{config,logger,paths}.py
config/config.yml
maps/default.yml
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for refactored project"
```

---

### Task G8: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite project section**

Remove all mention of: communication, NOMA, BER, OFDMA, channel.yml, scenario.npz. Replace with new structure (algorithms/, envs/, core/, api/). Keep language conventions (Chinese comments / English CLI output) and matplotlib unicode-minus tip.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for refactored project structure"
```

---

## Phase H — Final verification

### Task H1: Full test suite

- [ ] **Step 1: Run** `pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Smoke-train each algorithm**

```bash
for algo in dqn madqn shared_madqn qmix vdn ppo mappo; do
  python main.py --algo $algo --mode train --num_episodes 3
done
```

Expected: each completes without error.

- [ ] **Step 3: API end-to-end**

```bash
uvicorn api.main:app --port 8000 &
sleep 2
curl -X POST http://localhost:8000/train \
     -H "Content-Type: application/json" \
     -d '{"algorithm":"dqn","map_file":"default","config_overrides":{"algorithm":{"num_episodes":2,"episode_length":20}}}'
# Note returned run_id, then:
sleep 5
curl http://localhost:8000/status/<run_id>
kill %1
```

Expected: status transitions running → completed.

- [ ] **Step 4: No commit** (verification only)

---

## Self-Review

After writing all task code, run this checklist:

1. **Spec coverage:** every section in `docs/superpowers/specs/2026-06-02-pure-algorithm-refactor-design.md` (§2 layout, §3 env, §4 algorithms, §5 core, §6 API, §7 config, §8 migration, §9 testing) maps to phase A–H above. ✓
2. **Placeholder scan:** No "TBD", "TODO", "implement later" markers; all code blocks contain runnable code. (Exceptions: per-step trainer extension in E1 is described in pseudo + concrete helper functions — clear enough.)
3. **Type consistency:** `BaseAlgorithm.take_action` signature `(states: dict[int, ...], explore: bool) -> dict[int, int]` is used consistently in DQN/MADQN definitions. Buffer kind strings (`"single"|"per_agent"|"joint"|"none"`) are used consistently across base/trainer.
4. **Known limitation surfaced:** API status only updates at end-of-training (no real-time callback). Documented in Task F2.

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-02-pure-algorithm-refactor.md`.**
