"""End-to-end check on the toy MIND data: preprocessing, graph, one forward pass, evaluation, API."""
import os
import subprocess
import sys

import numpy as np
import pytest

from src.config import ROOT, load_config

TEST_CFG = "configs/test.yaml"


@pytest.fixture(scope="module")
def built():
    # Toy data and artefacts live in data/toy/, so the real MIND files are never touched
    subprocess.run([sys.executable, "scripts/make_toy_data.py", "data/toy/raw"], cwd=ROOT, check=True)
    for module in ("src.data.preprocess", "src.data.db", "src.graph.build_graph"):
        subprocess.run([sys.executable, "-m", module, TEST_CFG], cwd=ROOT, check=True)
    os.environ["RECSYS_CONFIG"] = TEST_CFG
    return load_config(TEST_CFG)


def test_forward_pass(built):
    import pandas as pd
    from src.graph.retrieval import GraphRecommender
    rec = GraphRecommender.from_config(built)
    row = pd.read_pickle(built["processed_dir"] / "behaviors_test.pkl").iloc[0]
    scores, paths, profile = rec.forward(row.history, row.candidates, rec.ignored.get(row.user_id))
    assert len(scores) == len(row.candidates) and np.isfinite(scores).all()
    assert len(paths) == len(row.candidates)


def test_eval_harness_on_mock_predictions(built):
    import pandas as pd
    from src.eval.evaluate import evaluate, random_model
    beh = pd.read_pickle(built["processed_dir"] / "behaviors_val.pkl")
    res = evaluate(beh, random_model(0))
    assert 0.3 < res["all"]["auc"] < 0.7 and "ndcg@10" in res["all"] and "recall@20" in res["all"]


def test_api_mock(built):
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    items = c.get("/recommendations", params={"user_id": "U1", "k": 5}).json()["items"]
    assert len(items) == 5
    r = c.post("/log_interaction", json={"user_id": "U1", "article_id": items[0]["article_id"],
                                         "feedback": "like"})
    assert r.json()["status"] == "stored"
    assert c.post("/log_interaction", json={"user_id": "U1", "article_id": "nope",
                                            "feedback": "like"}).status_code == 404
