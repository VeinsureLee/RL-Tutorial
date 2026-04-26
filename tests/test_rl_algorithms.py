"""单元测试：src/rl_algorithms/。"""
import math
import os

import numpy as np
import pytest


# ---------------------------------------------------------------- ReplayBuffer

def test_replay_buffer_fifo_and_sample():
    from rl_algorithms.replay import ReplayBuffer
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(i, i * 10, float(i), i + 1, False)
    # 容量 3：最后 3 条留下
    assert len(buf) == 3
    batch = buf.sample(3)
    # 只验证结构，不验证顺序
    states, actions, rewards, next_states, dones = batch
    assert len(states) == 3
    assert set(states).issubset({2, 3, 4})


# ---------------------------------------------------------------- run_manager

def test_run_context_name_format(tmp_path, monkeypatch):
    """RunContext.new 产生的 name 符合预期格式。"""
    from utils import run_manager
    # 把 experiments/runs 指向 tmp_path
    monkeypatch.setattr(run_manager, "get_abs_path",
                        lambda rel: str(tmp_path / rel))

    rc = run_manager.RunContext.new(
        algo="madqn", num_agents=4,
        extra={"omega": 1.0, "lr": 1e-4},
        tag="smoke",
        timestamp="20260425_1030",
    )
    assert rc.name == "20260425_1030_madqn_K4_omega1_lr1e-04_smoke"
    # 目录已创建
    import os
    assert os.path.isdir(rc.dir)
    assert os.path.isdir(rc.figs_dir)
    assert os.path.isdir(rc.test_dir)


def test_run_context_metrics_csv(tmp_path, monkeypatch):
    """write_metrics_csv 生成的表头 / 行数正确。"""
    from utils import run_manager
    monkeypatch.setattr(run_manager, "get_abs_path",
                        lambda rel: str(tmp_path / rel))

    rc = run_manager.RunContext.new(algo="dqn", num_agents=2,
                                    timestamp="20260425_1100")
    history = {
        "return_list": [1.0, 2.0, 3.0],
        "step_return_list": [-1, -1, -1],
        "approach_return_list": [2, 3, 4],
        "comm_return_list": [0, 0, 0],
        "ber_list": [5.0, 6.0, 7.0],
        "agent_return_lists": [[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]],
        "agent_step_return_lists": [[-0.5]*3, [-0.5]*3],
        "agent_approach_return_lists": [[1]*3, [1]*3],
        "agent_comm_return_lists": [[0]*3, [0]*3],
        "agent_ber_lists": [[5]*3, [5]*3],
    }
    rc.write_metrics_csv(history)
    with open(rc.metrics_path, encoding="utf-8") as f:
        lines = f.readlines()
    # 1 行表头 + 3 行数据
    assert len(lines) == 4
    header = lines[0].strip().split(",")
    assert "episode" in header
    assert "return" in header
    assert "agent0_return" in header
    assert "agent1_return" in header


# ---------------------------------------------------------------- Qnet 构建

def test_qnet_forward_shape(env_cfg):
    """Qnet 对单个状态能输出 (batch, n_actions) 形状。"""
    import torch
    from env.env import MultiRobotEnv
    from rl_algorithms.qnet_dqn import Qnet

    env = MultiRobotEnv(env_cfg)
    net = Qnet(
        state_num=env.n_states,
        action_dim=env.n_actions,
        rows=env.rows, cols=env.cols,
        embedding_dim=16,
        hidden_dim=32,
    )
    s = torch.tensor([env.pos_to_index(0, 0)], dtype=torch.long)
    t = torch.tensor([env.pos_to_index(*env.target_states[0])], dtype=torch.long)
    with torch.no_grad():
        q = net(s, t)
    assert q.shape == (1, env.n_actions)
    assert torch.isfinite(q).all()


# ---------------------------------------------------------------- 集成测试
# 思路：把 take_action 替换成均匀随机，让 trainer 完整跑一遍 episodes，断言 history /
# figs / metrics.csv 全套产物正确。算法是否能"学" 不在本测试范围内。

def test_random_policy_train_pipeline(env_cfg, tmp_path, monkeypatch):
    """随机策略走完 trainer 流水线：history 字段齐全 + 数值有限 + 图与 CSV 落盘。"""
    import torch
    from env.env import MultiRobotEnv
    from rl_algorithms.madqn import MADQN
    from rl_algorithms.trainer import train
    from rl_algorithms.plot import plot_training
    from utils import run_manager
    from utils.run_manager import RunContext

    # 把 RunContext 的 experiments/runs 重定向到 tmp_path，避免污染真实产物目录
    monkeypatch.setattr(run_manager, "get_abs_path",
                        lambda rel: str(tmp_path / rel))

    env = MultiRobotEnv(env_cfg)

    # MADQN + epsilon=1.0 让 take_action 永远走随机分支 = 真随机策略
    model = MADQN(
        env,
        lr=1e-4, gamma=0.99,
        epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0,
        hidden_dim=16, update_freq=4,
        replay_buffer_size=64,
        device=torch.device("cpu"),
    )

    num_eps = 3
    history = train(
        env=env, model=model,
        num_iterations=1, num_episodes=num_eps,
        episode_length=5, batch_size=4, train_interval=1,
    )

    # 1) history 必备 key
    expected_keys = {
        "return_list", "step_return_list", "approach_return_list", "comm_return_list",
        "agent_return_lists", "agent_step_return_lists",
        "agent_approach_return_lists", "agent_comm_return_lists",
        "ber_list", "agent_ber_lists", "time_list",
    }
    missing = expected_keys - set(history.keys())
    assert not missing, f"history missing keys: {missing}"

    # 2) 全局序列长度 = num_episodes，每 agent 序列长度同样
    for k in ("return_list", "step_return_list", "approach_return_list",
              "comm_return_list", "ber_list", "time_list"):
        assert len(history[k]) == num_eps, f"{k} length {len(history[k])} != {num_eps}"
    for k in ("agent_return_lists", "agent_step_return_lists",
              "agent_approach_return_lists", "agent_comm_return_lists",
              "agent_ber_lists"):
        assert len(history[k]) == env.num_agents
        for ag_vals in history[k]:
            assert len(ag_vals) == num_eps

    # 3) 全列有限 / 非 NaN
    def _all_finite(seq):
        return all(math.isfinite(float(x)) for x in seq)

    for k in ("return_list", "step_return_list", "approach_return_list",
              "comm_return_list", "ber_list", "time_list"):
        assert _all_finite(history[k]), f"{k} has NaN/Inf"
    for k in ("agent_return_lists", "agent_step_return_lists",
              "agent_approach_return_lists", "agent_comm_return_lists",
              "agent_ber_lists"):
        for ag_vals in history[k]:
            assert _all_finite(ag_vals), f"{k} per-agent has NaN/Inf"

    # 4) plot_training 应落盘 7 张图（5 张原 metrics + path_reward + time.png）
    fig_dir = str(tmp_path / "figs")
    paths = plot_training(history, fig_dir=fig_dir, prefix="rand", algo_label="RAND")
    assert len(paths) == 7
    for p in paths:
        assert os.path.isfile(p), f"figure missing: {p}"
    assert any(p.endswith("time.png") for p in paths), "time.png not produced"
    assert any(p.endswith("path_reward.png") for p in paths), "path_reward.png not produced"

    # 5) RunContext.write_metrics_csv 落盘后没有 NaN/Inf 列
    rc = RunContext.new(
        algo="madqn", num_agents=env.num_agents,
        timestamp="20260426_9999", tag="random_pipeline_test",
    )
    csv_path = rc.write_metrics_csv(history)
    assert os.path.isfile(csv_path)
    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == num_eps + 1  # 1 表头 + N 行数据
    header = [h.strip() for h in lines[0].strip().split(",")]
    assert "time_sec" in header
    # 数据行：列数与表头一致 + 全部可解析为 float
    for line in lines[1:]:
        cells = line.strip().split(",")
        assert len(cells) == len(header)
        for cell in cells:
            v = float(cell)
            assert math.isfinite(v), f"NaN/Inf cell in metrics.csv: {cell!r}"
