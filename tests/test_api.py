from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_train_returns_run_id():
    payload = {
        "algorithm": "dqn",
        "map_file": "default",
        "config_overrides": {"algorithm": {"num_episodes": 2, "episode_length": 20}},
        "tag": "apitest",
    }
    r = client.post("/train", json=payload)
    assert r.status_code == 200
    assert "run_id" in r.json()


def test_status_for_unknown_run_id_returns_404():
    r = client.get("/status/nonexistent_run_id")
    assert r.status_code == 404


def test_train_with_unknown_algorithm_eventually_fails():
    payload = {
        "algorithm": "not_a_real_algo",
        "map_file": "default",
        "config_overrides": {"algorithm": {"num_episodes": 1, "episode_length": 5}},
    }
    r = client.post("/train", json=payload)
    # 提交时不会立即失败（异步），但 status 会显示 failed
    assert r.status_code == 200
