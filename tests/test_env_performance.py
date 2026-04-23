"""
env 模块性能测试。

给出典型操作的耗时数字 + 上限断言，便于跟踪 regression。
运行 ``pytest tests/test_env_performance.py -s`` 可以看到打印的 ms 数字。
通过 ``run_tests.py`` 运行时自动带 -s。

上限（ENV_STEP_MS_MAX 等）是经验值，走得太慢时主动失败；若某台机器本就慢，
可以按实际放宽。注意：首次构造 env 会触发 radio_map 预计算，耗时显著长于稳态
step，所以 ``test_radio_map_init_time`` 单独一项，不进入 step 的上限。
"""
import numpy as np
import pytest

from _helpers import (
    DIR_DOWN, DIR_UP, DIR_LEFT, DIR_RIGHT,
    make_custom_env,
    time_it, print_timing,
)

# ---- 性能上限（ms） -----------------------------------------------------
ENV_STEP_MS_MAX = 20.0         # 单步 env.step 稳态上限
ENV_RESET_MS_MAX = 5.0
COMPUTE_BER_MS_MAX = 15.0      # compute_ber_rewards 稳态（K=1 退化路径）
RADIO_MAP_CACHED_INIT_MS_MAX = 800.0   # 命中 .npz 缓存的情况
# 冷启动（cache miss）没有上限断言——只打印，因机器差异大


# ---- Fixtures -----------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    """K=1、干净场景的共享 env（模块级复用，避免重复预计算 radio_map）。"""
    return make_custom_env(start=(60, 30), target=(0, 0), forbidden_areas=[])


# ---- env.step -----------------------------------------------------------

def test_env_step_perf(env):
    """env.step 稳态耗时：应该 < 20ms。"""
    env.reset()
    actions = [0] * env.num_agents

    r = time_it("env.step (steady state)", lambda: env.step(actions))
    print_timing(r)

    assert r["mean_ms"] < ENV_STEP_MS_MAX, (
        f"env.step mean={r['mean_ms']:.2f}ms > budget {ENV_STEP_MS_MAX}ms"
    )


def test_env_step_varied_actions_perf(env):
    """轮流走四个方向的耗时——确保 forbidden/边界路径不比正常路径慢多少。"""
    env.reset()
    cycle = [DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP]
    idx = {"i": 0}
    def step():
        a = cycle[idx["i"] % 4] * env.n_powers
        idx["i"] += 1
        env.step([a])
    r = time_it("env.step (R/D/L/U cycle)", step)
    print_timing(r)
    assert r["mean_ms"] < ENV_STEP_MS_MAX


# ---- env.reset ----------------------------------------------------------

def test_env_reset_perf(env):
    """env.reset：只重置位置/轨迹/done/prev_ber，不重算 radio_map，应该 < 5ms。"""
    r = time_it("env.reset", env.reset)
    print_timing(r)
    assert r["mean_ms"] < ENV_RESET_MS_MAX, (
        f"env.reset mean={r['mean_ms']:.2f}ms > budget {ENV_RESET_MS_MAX}ms"
    )


# ---- compute_ber_rewards 隔离计时 --------------------------------------

def test_compute_ber_rewards_perf(env):
    """绕开 env.step，直接对 K=1 位置调 compute_ber_rewards。"""
    from communication.ber_reward import compute_ber_rewards
    env.reset()
    positions = np.array([env.positions[0]], dtype=int)
    power_actions = np.array([0], dtype=int)
    kwargs = dict(
        radio_map=env.radio_map,
        positions=positions,
        power_actions=power_actions,
        P_sum=env.P_sum,
        num_power_levels=env.n_powers,
        N=env.N, D=env.D,
        noise_power=env.noise_power,
        rng=env.rng,
    )
    r = time_it(
        "compute_ber_rewards (K=1)",
        lambda: compute_ber_rewards(**kwargs),
    )
    print_timing(r)
    assert r["mean_ms"] < COMPUTE_BER_MS_MAX, (
        f"compute_ber_rewards mean={r['mean_ms']:.2f}ms > budget {COMPUTE_BER_MS_MAX}ms"
    )


# ---- radio_map 初始化（冷/热）-----------------------------------------

def test_radio_map_cached_init_perf():
    """
    构造 env 时命中 radio_map_cache.npz 的情形：走 .npz 反序列化路径。
    第一次运行本仓库时 cache 可能不存在；让首次跑把它建好，之后所有 run 都走 cache。
    """
    import time
    t0 = time.perf_counter()
    env = make_custom_env(start=(60, 30), target=(0, 0), forbidden_areas=[])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    del env  # 显式释放
    print(f"\n  [PERF] env.__init__ (radio_map cache hit)  {elapsed_ms:.1f} ms")
    assert elapsed_ms < RADIO_MAP_CACHED_INIT_MS_MAX, (
        f"env init too slow when cached: {elapsed_ms:.1f}ms > {RADIO_MAP_CACHED_INIT_MS_MAX}ms. "
        "First run builds cache; re-run should be fast."
    )


# ---- 全回合吞吐量估算 ---------------------------------------------------

def test_env_throughput(env):
    """连续 500 步的总耗时，报告 steps/sec。不设硬上限（只打印）。"""
    env.reset()
    actions = [0] * env.num_agents
    import time
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        env.step(actions)
    elapsed = time.perf_counter() - t0
    sps = N / elapsed
    print(f"\n  [PERF] env.step throughput  {sps:.0f} steps/sec  ({N} steps in {elapsed*1000:.0f} ms)")
    # 作为粗健康检查：至少 > 50 steps/sec
    assert sps > 50, f"env.step throughput {sps:.0f} < 50 steps/sec"
