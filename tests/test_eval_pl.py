"""Tests of the evaluation tools (PL): metrics on toy data, ablations, test lock, leakage check."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.config import ROOT, load_config

TEST_CFG = "configs/test.yaml"


@pytest.fixture(scope="module")
def built():
    subprocess.run([sys.executable, "scripts/make_toy_data.py", "data/toy/raw"], cwd=ROOT, check=True)
    for module in ("src.data.preprocess", "src.data.db", "src.graph.build_graph"):
        subprocess.run([sys.executable, "-m", module, TEST_CFG], cwd=ROOT, check=True)
    os.environ["RECSYS_CONFIG"] = TEST_CFG
    return load_config(TEST_CFG)


def test_ablations_and_slices(built):
    import pandas as pd
    from src.eval.evaluate import evaluate, graph_model, graph_variant
    from src.graph.retrieval import GraphRecommender
    rec = GraphRecommender.from_config(built)
    beh = pd.read_pickle(built["processed_dir"] / "behaviors_val.pkl").head(100)
    for fn in (graph_model(rec), graph_variant(rec, ignored_weight=0.0), graph_variant(rec, hops=1)):
        res = evaluate(beh, fn, popularity=rec.popularity, category=rec.category)
        assert {"ndcg@10", "recall@20", "top10_mean_popularity", "top10_n_categories"} <= set(res["all"])
    assert rec.ignored_weight == built["retrieval"]["ignored_weight"]     # the variant is a copy


def test_leakage_check_passes(built):
    from src.eval.leakage_check import main
    assert main(TEST_CFG)


def test_test_split_needs_final_flag():
    from src.eval.evaluate import guard_test_split
    with pytest.raises(SystemExit):
        guard_test_split(SimpleNamespace(final=False, force=False))


def test_failed_test_run_is_not_logged(built, monkeypatch, tmp_path):
    import src.eval.evaluate as ev
    monkeypatch.setattr(ev, "TEST_LOG", tmp_path / "log.json")
    args = SimpleNamespace(final=True, force=False, models=["graph"], sample=0, seeds=[1])
    ev.guard_test_split(args)                       # check only: nothing written
    assert not (tmp_path / "log.json").exists()
    ev.record_test_run(args, tmp_path / "eval_test_x.json")
    with pytest.raises(SystemExit):                 # second run refused once one has completed
        ev.guard_test_split(args)


def test_one_score_call_per_impression(built):
    import pandas as pd
    from src.eval.evaluate import evaluate
    beh = pd.read_pickle(built["processed_dir"] / "behaviors_val.pkl").head(20)
    calls = []
    evaluate(beh, lambda row: calls.append(1) or [0.0] * len(row.candidates))
    assert len(calls) == len(beh)
