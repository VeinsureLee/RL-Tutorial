"""单元测试：src/rl_algorithms/。"""
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
    from rl_algorithms.qnet import Qnet

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
