from envs.map_builder import WALL, build_grid_array, load_map_spec


def test_load_map_spec_returns_dict():
    spec = load_map_spec("default")
    assert isinstance(spec["size"], list) and len(spec["size"]) == 2
    assert spec["num_agents"] == len(spec["agents_start"])
    assert len(spec["rooms"]) >= 2
    assert len(spec["doors"]) >= 1


def test_build_grid_array_shape():
    spec = load_map_spec("default")
    grid = build_grid_array(spec)
    assert grid.shape == tuple(spec["size"])


def test_build_grid_array_has_walls():
    spec = load_map_spec("default")
    grid = build_grid_array(spec)
    # 至少有部分墙体存在
    assert (grid == WALL).sum() > 0


def test_build_grid_array_has_doors_as_open():
    spec = load_map_spec("default")
    grid = build_grid_array(spec)
    for door in spec["doors"]:
        r, c = door["position"]
        assert grid[r, c] != WALL, f"Door at ({r}, {c}) is blocked"
