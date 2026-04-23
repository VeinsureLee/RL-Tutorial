"""
env 模块可视化测试：给 env 一串脚本动作，看轨迹图是否与预期一致。

每个 test 会：
1. 在一个 K=1 的干净 env 上重置到指定起点
2. 跑一段动作序列
3. 断言关键位置 / 终点 / 步数
4. 保存一张轨迹 PNG 到 ``experiments/unit_tests/<today>/viz/``

pytest 通过 = 行为层面正确；PNG 用于肉眼验证（有时 trajectory 对齐但边界处理
不符合期望，看图最直观）。

Run:  VSCode 里 Run Python File run_tests.py  |  或 pytest tests/test_env_visualization.py -s
"""
import pytest

from _helpers import (
    DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_STAY,
    make_custom_env, override_forbidden_set,
    run_scripted, save_trajectory_fig, viz_out_dir,
)


# ---- Fixture：共享 env（避免重复算 radio_map）---------------------------

@pytest.fixture(scope="module")
def env():
    """K=1，无禁区，开放场地。每个测试内部用 start/target 覆盖与轨迹验证。"""
    return make_custom_env(start=(60, 30), target=(0, 0), forbidden_areas=[])


@pytest.fixture(scope="module", autouse=True)
def _announce_viz_dir():
    """模块开始时打印 viz 输出位置。"""
    d = viz_out_dir()
    print(f"\n  [VIZ] saving trajectory PNGs to: {d}")
    yield
    print(f"\n  [VIZ] done, open {d} to view.")


# ---- 1. 直线上 10 ------------------------------------------------------

def test_viz_up_10(env):
    """从 (30, 30) 向上走 10 步，终点应为 (20, 30)。"""
    env.start_states = [(30, 30)]
    env.target_states = [(0, 0)]
    res = run_scripted(env, [DIR_UP] * 10)

    expected_end = (20, 30)
    assert res["positions"][-1] == expected_end, (
        f"expected end {expected_end}, got {res['positions'][-1]}"
    )
    assert res["positions"][0] == (30, 30)
    assert len(res["positions"]) == 11  # 起点 + 10 步
    # 每步都是 closer（起点比终点接近 target=(0,0) 的上方）
    # 断言 10 步都是 reward_closer，不因为边界 / 禁区走形
    assert all(ar == env.reward_closer for ar in res["approach_rewards"]), \
        f"approach_rewards={res['approach_rewards']}"

    path = save_trajectory_fig(
        env,
        title="Test: UP x 10 from (30,30)",
        subtitle=f"end={res['positions'][-1]}, expected {expected_end}",
        filename="up10.png",
    )
    print(f"\n  saved: {path}")


# ---- 2. 直线下 10 ------------------------------------------------------

def test_viz_down_10(env):
    """从 (30, 30) 向下走 10 步，终点 (40, 30)。"""
    env.start_states = [(30, 30)]
    env.target_states = [(119, 59)]  # 右下角外围，避免意外到达
    res = run_scripted(env, [DIR_DOWN] * 10)

    expected_end = (40, 30)
    assert res["positions"][-1] == expected_end
    path = save_trajectory_fig(env,
        title="Test: DOWN x 10 from (30,30)",
        subtitle=f"end={res['positions'][-1]}",
        filename="down10.png")
    print(f"\n  saved: {path}")


# ---- 3. 直线右 15 / 左 15 -----------------------------------------------

def test_viz_right_15(env):
    env.start_states = [(60, 10)]
    env.target_states = [(0, 0)]
    res = run_scripted(env, [DIR_RIGHT] * 15)
    assert res["positions"][-1] == (60, 25)
    path = save_trajectory_fig(env,
        title="Test: RIGHT x 15 from (60,10)",
        subtitle=f"end={res['positions'][-1]}",
        filename="right15.png")
    print(f"\n  saved: {path}")


def test_viz_left_15(env):
    env.start_states = [(60, 45)]
    env.target_states = [(0, 0)]
    res = run_scripted(env, [DIR_LEFT] * 15)
    assert res["positions"][-1] == (60, 30)
    path = save_trajectory_fig(env,
        title="Test: LEFT x 15 from (60,45)",
        subtitle=f"end={res['positions'][-1]}",
        filename="left15.png")
    print(f"\n  saved: {path}")


# ---- 4. L 形：右 10 后下 10 --------------------------------------------

def test_viz_L_shape(env):
    """先右 10 再下 10，形成 L 形。"""
    env.start_states = [(30, 20)]
    env.target_states = [(119, 59)]
    actions = [DIR_RIGHT] * 10 + [DIR_DOWN] * 10
    res = run_scripted(env, actions)

    assert res["positions"][10] == (30, 30)   # 拐点
    assert res["positions"][-1] == (40, 30)    # 终点

    path = save_trajectory_fig(env,
        title="Test: L-shape (right 10 + down 10)",
        subtitle=f"corner=(30,30) end=(40,30)",
        filename="L_shape.png")
    print(f"\n  saved: {path}")


# ---- 5. 正方形回路 -----------------------------------------------------

def test_viz_square_loop(env):
    """右 5 下 5 左 5 上 5，回到起点 (30, 30)。"""
    env.start_states = [(30, 30)]
    env.target_states = [(119, 59)]
    actions = ([DIR_RIGHT] * 5 + [DIR_DOWN] * 5
               + [DIR_LEFT] * 5 + [DIR_UP] * 5)
    res = run_scripted(env, actions)

    # 四个角 + 回到起点
    corners = [(30, 35), (35, 35), (35, 30), (30, 30)]
    for step_idx, expected in zip([5, 10, 15, 20], corners):
        got = res["positions"][step_idx]
        assert got == expected, f"step {step_idx}: {got} != {expected}"

    path = save_trajectory_fig(env,
        title="Test: square loop (R5 D5 L5 U5)",
        subtitle="should return to (30,30)",
        filename="square_loop.png")
    print(f"\n  saved: {path}")


# ---- 6. 撞地图上边界 --------------------------------------------------

def test_viz_hit_top_boundary(env):
    """起点 (2, 30) 向上走 5 步：2 步后抵 row=0，后 3 步应该越界回退，位置保持 (0, 30)。"""
    env.start_states = [(2, 30)]
    env.target_states = [(119, 59)]
    res = run_scripted(env, [DIR_UP] * 5)

    assert res["positions"][0] == (2, 30)
    assert res["positions"][1] == (1, 30)
    assert res["positions"][2] == (0, 30)
    # 之后 3 步都被边界回退
    for i in (3, 4, 5):
        assert res["positions"][i] == (0, 30), \
            f"step {i}: expected stay at (0,30), got {res['positions'][i]}"
    # 最后 3 步应该有 reward_forbidden（越界按 forbidden 处理）
    assert res["approach_rewards"][-1] == env.reward_forbidden

    path = save_trajectory_fig(env,
        title="Test: hit top boundary (UP x 5 from (2,30))",
        subtitle=f"final pos=(0,30), last reward={res['approach_rewards'][-1]}",
        filename="hit_top_boundary.png")
    print(f"\n  saved: {path}")


# ---- 7. 撞禁区 --------------------------------------------------------

def test_viz_hit_forbidden(env):
    """在 (20..22) × (20..22) 人工放一块禁区，从 (20, 18) 向右走 5 步应在 (20, 19) 停住。"""
    env.start_states = [(20, 18)]
    env.target_states = [(119, 59)]
    forbidden = {(r, c) for r in (20, 21, 22) for c in (20, 21, 22)}
    override_forbidden_set(env, forbidden)

    try:
        res = run_scripted(env, [DIR_RIGHT] * 5)

        assert res["positions"][0] == (20, 18)
        assert res["positions"][1] == (20, 19)
        # (20, 20) 是禁区，后 4 步应全停 (20, 19)
        for i in (2, 3, 4, 5):
            assert res["positions"][i] == (20, 19), \
                f"step {i}: expected (20,19), got {res['positions'][i]}"
        assert res["approach_rewards"][-1] == env.reward_forbidden

        path = save_trajectory_fig(env,
            title="Test: hit forbidden region (RIGHT x 5 from (20,18))",
            subtitle="forbidden block at (20..22, 20..22), stuck at (20,19)",
            filename="hit_forbidden.png")
        print(f"\n  saved: {path}")
    finally:
        # 清理：恢复空禁区避免影响后续测试
        override_forbidden_set(env, [])


# ---- 8. 到达目标即 done -----------------------------------------------

def test_viz_reach_target(env):
    """target = (30, 35)，start = (30, 30)，右 5 应第 5 步 done。"""
    env.start_states = [(30, 30)]
    env.target_states = [(30, 35)]
    res = run_scripted(env, [DIR_RIGHT] * 8)  # 特意多跑 3 步

    assert res["dones_when"] == 5, f"expected done at step 5, got {res['dones_when']}"
    assert res["positions"][5] == (30, 35)
    # 第 5 步的 approach_reward 应该是 reward_goal
    assert res["approach_rewards"][4] == env.reward_goal, \
        f"approach_reward at step 5 = {res['approach_rewards'][4]}, expected reward_goal={env.reward_goal}"

    path = save_trajectory_fig(env,
        title="Test: reach target (RIGHT x 8, target=(30,35))",
        subtitle=f"done at step {res['dones_when']}, reward_goal={env.reward_goal}",
        filename="reach_target.png")
    print(f"\n  saved: {path}")


# ---- 9. STAY 动作 -----------------------------------------------------

def test_viz_stay(env):
    """原地 STAY 5 次，位置不动，每步 approach_reward = reward_same。"""
    env.start_states = [(60, 30)]
    env.target_states = [(0, 0)]
    res = run_scripted(env, [DIR_STAY] * 5)

    for pos in res["positions"]:
        assert pos == (60, 30)
    for ar in res["approach_rewards"]:
        assert ar == env.reward_same, f"stay gave approach={ar} expected {env.reward_same}"

    path = save_trajectory_fig(env,
        title="Test: STAY x 5 from (60,30)",
        subtitle=f"pos unchanged, approach={env.reward_same} per step",
        filename="stay5.png")
    print(f"\n  saved: {path}")
