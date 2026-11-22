"""Tests of the MLE-Ops Week 3 parts on toy data with the mock LLM: client cost, summary, fusion layer."""
import numpy as np
import pandas as pd

from src.config import load_config
from tests.test_llm import built  # noqa: F401  (builds the toy data once)


def test_client_cache_and_cost(built, tmp_path):  # noqa: F811
    from src.llm.client import LLMClient
    c = LLMClient(provider="mock", cache_dir=tmp_path, price_input_per_mtok=1.0, price_output_per_mtok=5.0)
    c.complete("sys", '{"id": "N1"}')
    assert not c.last_call["cached"] and c.last_call["input_tokens"] > 0 and c.stats["cost_usd"] > 0
    c.complete("sys", '{"id": "N1"}')
    assert c.last_call["cached"] and c.stats["cache_hits"] == 1 and c.stats["calls"] == 1


def test_profile_summary(built):  # noqa: F811
    from src.graph.retrieval import GraphRecommender
    from src.llm.client import LLMClient
    from src.llm.profile_summary import summarize
    rec = GraphRecommender.from_config(built)
    row = pd.read_pickle(built["processed_dir"] / "behaviors_val.pkl").iloc[0]
    res = summarize(rec, rec.build_profile(row.history), LLMClient.from_config(built))
    assert res["status"] == "ok" and res["summary"]
    empty = summarize(rec, rec.build_profile([]), LLMClient.from_config(built))
    assert empty["status"] == "empty_profile"


def test_fusion_layer_trains_and_reranks(built):  # noqa: F811
    from src.graph.retrieval import GraphRecommender
    from src.llm.client import LLMClient
    from src.llm.fusion import FEATURES, FusionLayer, train_from_config
    from src.llm.rerank import Reranker
    layer, history, info = train_from_config(built, 150, 80, 1, epochs=5, log=lambda *_: None)
    assert len(layer.w) == len(FEATURES) and history[-1]["train_loss"] <= history[0]["train_loss"]
    rec = GraphRecommender.from_config(built)
    rr = Reranker.from_config(built, rec, LLMClient.from_config(built))
    rr.fusion = layer
    row = pd.read_pickle(built["processed_dir"] / "behaviors_val.pkl").iloc[0]
    profile = rec.build_profile(row.history)
    cands = list(row.candidates)
    res = rr.rerank(profile, cands, rec.score(profile, cands))
    assert sorted(res["order"]) == sorted(cands)
    assert isinstance(FusionLayer().score(np.zeros((3, len(FEATURES)))), np.ndarray)
