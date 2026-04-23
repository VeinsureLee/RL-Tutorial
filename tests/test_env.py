"""单元测试：src/env/。"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def env(env_cfg):
    from env.env import MultiRobotEnv
    return MultiRobotEnv(env_cfg)


def test_reset_returns_state_indices(env):
    states = env.reset()
    assert isinstance(states, list)
    assert len(states) == env.num_agents
    rows, cols = env.map_size
    for s in states:
        assert 0 <= s < rows * cols


def test_step_shapes(env):
    env.reset()
    actions = [0] * env.num_agents
    ns, rewards, dones, info = env.step(actions)
    assert len(ns) == env.num_agents
    assert len(rewards) == env.num_agents
    assert len(dones) == env.num_agents
    for key in ("ber", "sinr", "power_indices",
                "step_rewards", "approach_rewards", "comm_rewards"):
        assert key in info
        assert len(info[key]) == env.num_agents


def test_reward_decomposition_sums_to_total(env):
    """每个活跃 agent 的 reward == step + approach + comm（允许浮点误差）。"""
    env.reset()
    actions = [1] * env.num_agents
    _, rewards, _, info = env.step(actions)
    for i in range(env.num_agents):
        total = info["step_rewards"][i] + info["approach_rewards"][i] + info["comm_rewards"][i]
        assert np.isclose(total, rewards[i], atol=1e-9), \
            f"agent {i}: total={total} vs r={rewards[i]}"


def test_stay_action_uses_reward_same(env):
    """对 start != target 的 agent 执行 STAY，approach_reward 应该是 reward_same。"""
    env.reset()
    stay_idx = env.directions.index((0, 0))
    actions = [stay_idx * env.n_powers] * env.num_agents
    _, _, _, info = env.step(actions)
    for i in range(env.num_agents):
        start = tuple(env.start_states[i])
        target = tuple(env.target_states[i])
        if start != target:
            assert info["approach_rewards"][i] == env.reward_same, \
                f"agent {i}: approach_reward={info['approach_rewards'][i]}"


def test_first_step_comm_reward_zero(env):
    """prev_ber 在 reset 后是 NaN，首步 comm_reward 必须是 0。"""
    env.reset()
    actions = [0] * env.num_agents
    _, _, _, info = env.step(actions)
    assert np.allclose(info["comm_rewards"], 0.0)


def test_pos_index_roundtrip(env):
    for r, c in [(0, 0), (5, 3), (env.rows - 1, env.cols - 1)]:
        idx = env.pos_to_index(r, c)
        assert env.index_to_pos(idx) == (r, c)


def test_decode_action_consistency(env):
    """decode_action(a) == (a // n_powers, a % n_powers)。"""
    for a in range(env.n_actions):
        d, p = env.decode_action(a)
        assert d == a // env.n_powers
        assert p == a % env.n_powers
        assert 0 <= d < env.n_dirs
        assert 0 <= p < env.n_powers
