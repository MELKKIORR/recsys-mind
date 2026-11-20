"""API in graph mode on toy data (SE): LLM re-ranking from the cache, likes stored and used, onboarding."""
import os
import shutil
import subprocess
import sys

import pytest

from src.config import ROOT, load_config

TEST_CFG = "configs/test.yaml"


@pytest.fixture(scope="module")
def client():
    subprocess.run([sys.executable, "scripts/make_toy_data.py", "data/toy/raw"], cwd=ROOT, check=True)
    for module in ("src.data.preprocess", "src.data.db", "src.graph.build_graph"):
        subprocess.run([sys.executable, "-m", module, TEST_CFG], cwd=ROOT, check=True)
    cfg = load_config(TEST_CFG)
    shutil.rmtree(cfg["llm"]["cache_dir"], ignore_errors=True)    # start with an empty LLM cache
    os.environ["RECSYS_CONFIG"] = TEST_CFG
    os.environ["PREDICT_MODE"] = "graph"
    from fastapi.testclient import TestClient
    from api.main import app, predictor
    predictor.cache_clear()
    with TestClient(app) as c:
        yield c


def reader(client):
    return client.get("/users/sample", params={"n": 1}).json()[0]["user_id"]


def test_graph_mode_no_mock(client):
    h = client.get("/health").json()
    assert h["mode"] == "graph" and h["llm"]["cache_only"] is True


def test_llm_rerank_served_from_cache(client):
    uid = reader(client)
    first = client.get("/recommendations", params={"user_id": uid, "k": 10})
    assert first.json()["llm_status"] == "fallback_cache_miss"      # graph order, cache filled after
    assert "X-Response-Time-ms" in first.headers
    second = client.get("/recommendations", params={"user_id": uid, "k": 10}).json()
    assert second["llm_status"] == "ok"
    assert [i["article_id"] for i in first.json()["items"]] and len(second["items"]) == 10
    assert any(i["evidence_path"] for i in second["items"])


def test_like_is_stored_and_changes_recommendations(client):
    uid = reader(client)
    before = client.get("/recommendations", params={"user_id": uid, "k": 10}).json()["items"]
    liked = before[0]["article_id"]
    r = client.post("/log_interaction", json={"user_id": uid, "article_id": liked, "feedback": "like"})
    assert r.json()["status"] == "stored"
    hist = client.get("/user_history", params={"user_id": uid}).json()["reactions"]
    assert hist[0]["article_id"] == liked and hist[0]["feedback"] == "like"
    after = client.get("/recommendations", params={"user_id": uid, "k": 10}).json()["items"]
    assert liked not in [i["article_id"] for i in after]


def test_onboarding_with_three_topics(client):
    cats = [c["category"] for c in client.get("/categories").json()][:3]
    assert len(cats) == 3
    assert client.post("/users", json={"topics": cats[:2]}).status_code == 422
    uid = client.post("/users", json={"topics": cats}).json()["user_id"]
    items = client.get("/recommendations", params={"user_id": uid, "k": 10}).json()["items"]
    assert items and all(i["reason"] != "Popular this week" for i in items)


def test_latency_endpoint(client):
    lat = client.get("/latency").json()
    assert lat["/recommendations"]["n"] >= 1 and lat["/recommendations"]["max"] > 0
