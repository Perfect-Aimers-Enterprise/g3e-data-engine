from fastapi.testclient import TestClient

from g3e_data_engine.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_classes():
    resp = client.get("/classes")
    assert resp.status_code == 200
    data = resp.json()
    names = [c["name"] for c in data["classes"]]
    assert names == ["person", "fire", "gun", "smoke", "knife", "car", "dog", "cat"]


def test_allocate_endpoint():
    resp = client.post("/pipeline/allocate", json={"total_images": 1000})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requested"] == 1000
    budgets = {b["class_name"]: b["target_images"] for b in data["budgets"]}
    assert budgets["person"] >= budgets["dog"]


def test_allocate_endpoint_with_overrides():
    resp = client.post(
        "/pipeline/allocate",
        json={"total_images": 1000, "overrides": {"smoke": 5.0}},
    )
    assert resp.status_code == 200
    data = resp.json()
    budgets = {b["class_name"]: b["target_images"] for b in data["budgets"]}

    baseline = client.post("/pipeline/allocate", json={"total_images": 1000}).json()
    baseline_budgets = {b["class_name"]: b["target_images"] for b in baseline["budgets"]}

    assert budgets["smoke"] >= baseline_budgets["smoke"]


def test_run_pipeline_dry_run_never_downloads():
    resp = client.post("/pipeline/run", json={"dry_run": True, "total_images": 500})
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is True
    assert data["accepted_images"] is None


def test_run_pipeline_real_run_refuses_cleanly_when_sources_not_ready():
    """weapons/fire_smoke ship with license.verified=False — must be a clean
    422, not a 500 traceback, and must happen before any download."""
    resp = client.post("/pipeline/run", json={"dry_run": False, "total_images": 500})
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "source_not_ready"
    assert "weapons" in data["detail"]
