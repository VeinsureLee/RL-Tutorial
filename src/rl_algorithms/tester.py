"""
测试接口：跑一个 episode，统计各类 reward/BER，生成 nav/signal GIF 与 PNG。
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from rl_algorithms.madqn import MADQN
from rl_algorithms.qmix import QMIX


def _take_all_actions(env, model, states):
    if isinstance(model, (MADQN, QMIX)):
        return model.take_action(states, training=False)
    actions = [int(np.random.randint(env.n_actions)) for _ in range(env.num_agents)]
    actions[model.agent_id] = model.take_action(states[model.agent_id], training=False)
    return actions


def _run_episode(env, model, max_steps: int):
    states = env.reset()
    total = 0.0
    step_sum = 0.0
    approach_sum = 0.0
    comm_sum = 0.0
    neglog_ber_all = []
    frames = [(
        env.positions.copy(),
        env.done_flags.copy(),
        [list(t) for t in env.trajectories],
    )]

    for step in range(max_steps):
        actions = _take_all_actions(env, model, states)
        next_states, rewards, dones, info = env.step(actions)
        total += float(sum(rewards))
        step_sum += float(np.sum(info["step_rewards"]))
        approach_sum += float(np.sum(info["approach_rewards"]))
        comm_sum += float(np.sum(info["comm_rewards"]))
        for b in info["ber"]:
            neglog_ber_all.append(-np.log10(max(float(b), 1e-20)))
        frames.append((
            env.positions.copy(),
            env.done_flags.copy(),
            [list(t) for t in env.trajectories],
        ))
        states = next_states
        if env.all_done:
            return True, step + 1, total, step_sum, approach_sum, comm_sum, neglog_ber_all, frames
    return False, max_steps, total, step_sum, approach_sum, comm_sum, neglog_ber_all, frames


def test(env, model, *, max_steps: int, out_dir: str, gif_fps: int = 5) -> dict:
    """
    跑一个测试 episode 并把所有产物写到 ``out_dir``（通常是 ``<run.dir>/test/``）。

    落盘文件名（无 prefix，每次 run 自有目录不会冲突）::
        out_dir/nav.gif
        out_dir/signal.gif
        out_dir/nav.png
        out_dir/signal.png
    """
    print("=" * 50)
    print(f"Testing agents={env.num_agents}, max_steps={max_steps}")
    print(f"out_dir={out_dir}")
    print("=" * 50)

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    success, steps, total, step_sum, appr, comm, neglog_list, frames = _run_episode(env, model, max_steps)
    elapsed = time.time() - t0

    if success:
        print(f"All agents reached targets in {steps} steps")
    else:
        print(f"Reached max_steps={max_steps}, not all agents done")
    print(f"total_reward={total:.2f}  step={step_sum:.2f}  approach={appr:.2f}  comm={comm:.2f}")
    if neglog_list:
        print(f"mean(-log10 BER)={np.mean(neglog_list):.2f}")
    print(f"elapsed={elapsed:.2f}s")

    artifacts = {}

    nav_gif = os.path.join(out_dir, "nav.gif")
    env.save_nav_gif(nav_gif, frames, fps=gif_fps)
    artifacts["nav_gif"] = nav_gif
    print(f"saved: {nav_gif}")

    sig_gif = os.path.join(out_dir, "signal.gif")
    env.save_signal_gif(sig_gif, frames, fps=gif_fps)
    artifacts["signal_gif"] = sig_gif
    print(f"saved: {sig_gif}")

    nav_png = os.path.join(out_dir, "nav.png")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    env.render_nav_frame(ax)
    fig.savefig(nav_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    artifacts["nav_png"] = nav_png
    print(f"saved: {nav_png}")

    sig_png = os.path.join(out_dir, "signal.png")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    env.render_signal_frame(ax)
    fig.savefig(sig_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    artifacts["signal_png"] = sig_png
    print(f"saved: {sig_png}")

    return {
        "success": success,
        "steps": steps,
        "total_reward": total,
        "step_reward": step_sum,
        "approach_reward": appr,
        "comm_reward": comm,
        "mean_neg_log_ber": float(np.mean(neglog_list)) if neglog_list else 0.0,
        "elapsed": elapsed,
        "artifacts": artifacts,
    }
