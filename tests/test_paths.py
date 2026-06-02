from utils.paths import config_path, map_path, project_root


def test_project_root_contains_pyproject():
    assert (project_root() / "pyproject.toml").exists()


def test_config_path_default():
    assert config_path().name == "config.yml"


def test_map_path_by_name():
    p = map_path("default")
    assert p.name == "default.yml"
    assert "maps" in str(p).replace("\\", "/")
