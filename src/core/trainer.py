"""统一训练循环。

通过 algo.required_buffer() 选择缓冲区类型；通过 algo.is_on_policy 选择
on/off-policy 执行分支。on-policy 分支在 PPO 任务中扩展。
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.base import BaseAlgorithm
from core.replay import JointReplayBuffer, ReplayBuffer


@dataclass
class TrainResult:
    run_id: str
    model_path: str
    history: list[dict] = field(default_factory=list)


# ---------- buffer helpers ----------

def _make_buffer(algo: BaseAlgorithm, env, algo_cfg: dict) -> Any:
    kind = algo.required_buffer()
    cap = algo_cfg["replay_buffer_size"]
    if kind == "single":
        return ReplayBuffer(cap)
    if kind == "per_agent":
        return {i: ReplayBuffer(cap) for i in range(env.num_agents)}
    if kind == "joint":
        return JointReplayBuffer(cap, env.num_agents, env.observation_space.shape[0])
    return None


def _push(buffer, kind: str, algo, states, actions, rewards, next_states, dones) -> None:
    if kind == "single":
        a = algo.controlled_agent
        buffer.push(states[a], actions[a], rewards[a], next_states[a], dones[a])
    elif kind == "per_agent":
        for i in buffer:
            buffer[i].push(states[i], actions[i], rewards[i], next_states[i], dones[i])
    elif kind == "joint":
        buffer.push(states, actions, rewards, next_states, dones)


def _sample(buffer, kind: str, batch_size: int):
    if kind in ("single", "joint"):
        return buffer.sample(batch_size)
    if kind == "per_agent":
        return {i: buffer[i].sample(batch_size) for i in buffer}
    return None


def _ready(buffer, kind: str, batch_size: int) -> bool:
    if kind in ("single", "joint"):
        return len(buffer) >= batch_size
    if kind == "per_agent":
        return all(len(b) >= batch_size for b in buffer.values())
    return False


# ---------- off-policy episode ----------

def _run_off_policy_episode(
    algo: BaseAlgorithm,
    env,
    buffer,
    kind: str,
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
        total_reward += sum(rewards.values())

        _push(buffer, kind, algo, states, actions, rewards, next_states, dones)
        states = next_states

        if learn and _ready(buffer, kind, batch_size) and step % train_interval == 0:
            batch = _sample(buffer, kind, batch_size)
            info = algo.update(batch)
            losses.append(info.get("loss", 0.0))

        if all(dones.values()):
            break

    return {
        "reward": total_reward,
        "steps": step_count,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }


def _train_off_policy(
    algo: BaseAlgorithm, env, cfg: dict, run_dir: Path
) -> TrainResult:
    algo_cfg = cfg["algorithm"]
    num_episodes = algo_cfg["num_episodes"]
    episode_length = algo_cfg["episode_length"]
    batch_size = algo_cfg["batch_size"]
    train_interval = algo_cfg["train_interval"]
    save_interval = cfg["logging"]["save_interval"]

    kind = algo.required_buffer()
    buffer = _make_buffer(algo, env, algo_cfg)
    history: list[dict] = []
    model_path = str(run_dir / "model.pth")

    for ep in range(num_episodes):
        t0 = time.time()
        info = _run_off_policy_episode(
            algo, env, buffer, kind, episode_length, train_interval, batch_size, True
        )
        info["episode"] = ep
        info["wall_time"] = time.time() - t0
        history.append(info)

        if ep % save_interval == 0 or ep == num_episodes - 1:
            algo.save(model_path)

    return TrainResult(run_id=run_dir.name, model_path=model_path, history=history)


# ---------- on-policy training (extended when PPO is migrated) ----------

def _train_on_policy(
    algo: BaseAlgorithm, env, cfg: dict, run_dir: Path
) -> TrainResult:
    """on-policy 训练分支（PPO/MAPPO 任务中实现）。当前为占位实现。"""
    raise NotImplementedError(
        "On-policy training not yet wired up. Will be added with PPO migration."
    )


def train(algo: BaseAlgorithm, env, cfg: dict, run_dir: Path) -> TrainResult:
    if algo.is_on_policy:
        return _train_on_policy(algo, env, cfg, run_dir)
    return _train_off_policy(algo, env, cfg, run_dir)
