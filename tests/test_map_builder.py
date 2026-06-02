from envs.map_builder import WALL, build_grid_array, load_map_spec


def test_load_map_spec_returns_dict():
    spec = load_map_spec("default")
    assert spec["size"] == [15, 15]
    assert spec["num_agents"] == 2
    assert len(spec["rooms"]) == 3
    assert len(spec["agents_start"]) == 2


def test_build_grid_array_shape():
    spec = load_map_spec("default")
    grid = build_grid_array(spec)
    assert grid.shape == (15, 15)


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
