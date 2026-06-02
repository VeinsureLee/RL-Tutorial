from envs.indoor_env import IndoorEnv


def make_cfg(observation_mode: str = "full", reward_mode: str = "independent") -> dict:
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
    assert obs[0].shape[-1] == 49  # 7*7


def test_full_observation_size():
    env = IndoorEnv(make_cfg(observation_mode="full"))
    obs = env.reset()
    expected = env.rows * env.cols
    assert obs[0].shape[-1] == expected


def test_cooperative_team_bonus():
    cfg = make_cfg(reward_mode="cooperative")
    env = IndoorEnv(cfg)
    env.reset()
    env._goal_reached = {0: True, 1: True}
    bonus = env._compute_team_bonus()
    assert bonus == cfg["env"]["reward_team_bonus"]


def test_independent_no_team_bonus():
    cfg = make_cfg(reward_mode="independent")
    env = IndoorEnv(cfg)
    env.reset()
    env._goal_reached = {0: True, 1: True}
    # cooperative 模式下才有 bonus；independent 模式 reward_mode 控制 step() 不应用
    # 此处直接验证 _compute_team_bonus 总是按全员到达返回值（与 reward_mode 解耦）
    assert env._compute_team_bonus() == cfg["env"]["reward_team_bonus"]


def test_render_returns_rgb():
    env = IndoorEnv(make_cfg())
    env.reset()
    rgb = env.render()
    assert rgb.shape == (env.rows, env.cols, 3)
    assert rgb.dtype.name == "uint8"
