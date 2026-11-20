"""Tests of the LLM re-ranking (MLE-Core) on toy data, with the mock LLM (no key, no network)."""
import subprocess
import sys

import numpy as np
import pytest

from src.config import ROOT, load_config

TEST_CFG = "configs/test.yaml"


@pytest.fixture(scope="module")
def built():
    subprocess.run([sys.executable, "scripts/make_toy_data.py", "data/toy/raw"], cwd=ROOT, check=True)
    for module in ("src.data.preprocess", "src.graph.build_graph"):
        subprocess.run([sys.executable, "-m", module, TEST_CFG], cwd=ROOT, check=True)
    return load_config(TEST_CFG)


def test_rerank_one_impression(built):
    import pandas as pd
    from src.graph.retrieval import GraphRecommender
    from src.llm.client import LLMClient
    from src.llm.rerank import Reranker
    rec = GraphRecommender.from_config(built)
    rr = Reranker.from_config(built, rec, LLMClient.from_config(built))
    row = pd.read_pickle(built["processed_dir"] / "behaviors_val.pkl").iloc[0]
    profile = rec.build_profile(row.history, rec.ignored.get(row.user_id, []))
    cands = list(row.candidates)
    res = rr.rerank(profile, cands, rec.score(profile, cands))
    assert sorted(res["order"]) == sorted(cands) and len(res["scores"]) == len(cands)
    assert rr.fallback_rate() == 0.0


def test_explanation_fidelity(built):
    import pandas as pd
    from src.eval.fidelity import explanation_fidelity
    from src.graph.retrieval import GraphRecommender
    rec = GraphRecommender.from_config(built)
    beh = pd.read_pickle(built["processed_dir"] / "behaviors_val.pkl").head(50)
    res = explanation_fidelity(rec, beh)
    assert 0 <= res["share_explained_that_drop"] <= 1
