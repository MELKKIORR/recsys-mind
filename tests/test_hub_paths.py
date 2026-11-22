"""Entities of the stop list are never used as evidence paths (MLE-Core)."""
import subprocess
import sys

import numpy as np
import pytest

from src.config import ROOT, load_config

TEST_CFG = "configs/test.yaml"


@pytest.fixture(scope="module")
def rec():
    subprocess.run([sys.executable, "scripts/make_toy_data.py", "data/toy/raw"], cwd=ROOT, check=True)
    for module in ("src.data.preprocess", "src.graph.build_graph"):
        subprocess.run([sys.executable, "-m", module, TEST_CFG], cwd=ROOT, check=True)
    from src.graph.retrieval import GraphRecommender
    return GraphRecommender.from_config(load_config(TEST_CFG))


def test_stop_list_from_config(rec):
    from src.graph.retrieval import GraphRecommender
    cfg = load_config(TEST_CFG)
    cfg["retrieval"]["path_stop_entities"] = ["Tesla"]
    rec = GraphRecommender.from_config(cfg)
    assert [l for l, b in zip(rec.ent_labels, rec.path_blocked) if b] == ["Tesla"]


def test_blocked_entity_never_in_a_path(rec):
    import pandas as pd
    beh = pd.read_pickle(load_config(TEST_CFG)["processed_dir"] / "behaviors_val.pkl").head(30)
    rec.path_blocked[:] = False
    rec.path_blocked[int(np.argmax(rec.entity_df))] = True         # block the most frequent entity
    blocked = rec.ent_labels[int(np.argmax(rec.entity_df))]
    for row in beh.itertuples(index=False):
        profile = rec.build_profile(row.history)
        for c in row.candidates:
            path = rec.evidence_path(profile, c)
            assert path is None or blocked not in path["entities"]
