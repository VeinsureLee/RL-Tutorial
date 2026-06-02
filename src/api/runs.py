"""训练任务的运行时注册表与异步后台执行。

内存中维护 {run_id: RunState}；POST /train 时启动后台任务。
重启服务后内存丢失（v1 接受此限制；持久化由网站层决定）。
"""
import asyncio
import traceback
from dataclasses import dataclass
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
    overrides = dict(overrides)
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
        def _do():
            env = IndoorEnv(cfg)
            algo = build_algorithm(cfg["algorithm"]["name"], env, cfg)
            return train(algo, env, cfg, run_dir)

        result = await loop.run_in_executor(None, _do)
        state.model_path = result.model_path
        if result.history:
            state.episode = result.history[-1]["episode"]
            state.latest_reward = result.history[-1]["reward"]
        state.status = "completed"
    except Exception as e:
        state.error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        state.status = "failed"
