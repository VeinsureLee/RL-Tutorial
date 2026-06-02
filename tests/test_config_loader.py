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
