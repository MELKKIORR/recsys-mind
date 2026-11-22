"""Duplicate articles (same title and abstract) are returned once, and offset never repeats one."""
from types import SimpleNamespace

from src.predict import Predictor


def make_predictor(mode):
    # a1/a2 and b1/b2 are duplicates (case and spaces differ); c and d are unique
    titles = {"a1": "Storm hits coast", "a2": "storm  hits COAST", "b1": "Election", "b2": "Election",
              "c": "Match report", "d": "New phone"}
    abstracts = {"a1": "Heavy rain.", "a2": "heavy rain.", "b1": "Results.", "b2": "Results.",
                 "c": "", "d": "Specs."}
    popularity = {"a1": 6, "a2": 5, "b1": 4, "b2": 3, "c": 2, "d": 1}
    p = object.__new__(Predictor)
    p.mode = mode
    p.cfg = {"rerank": {"max_per_category": 0}}
    p.pool = sorted(titles)
    p.known_history = {"u": ["h"]}
    p.abstract = abstracts
    p.rec = SimpleNamespace(
        title=titles, popularity=popularity, ignored={}, category={},
        build_profile=lambda history, ignored, topics: None,
        score=lambda profile, pool: [popularity[a] for a in pool],
        evidence_path=lambda profile, a: None)
    # Reranker that keeps the graph order, to check what reaches the LLM
    p.seen_by_llm = []

    def rerank(profile, shortlist, scores):
        p.seen_by_llm = list(shortlist)
        return {"order": list(shortlist), "paths": {}, "reasons": {}, "status": "ok"}
    p.reranker = p.live_reranker = SimpleNamespace(rerank=rerank)
    return p


def ids(recs):
    return [r["article_id"] for r in recs]


def test_mock_mode_returns_one_article_per_duplicate_group():
    p = make_predictor("mock")
    assert ids(p.predict("u", k=10)) == ["a1", "b1", "c", "d"]


def test_graph_mode_dedupes_before_the_llm():
    p = make_predictor("graph")
    assert ids(p.predict("u", k=10)) == ["a1", "b1", "c", "d"]
    assert p.seen_by_llm == ["a1", "b1", "c", "d"]


def test_offset_does_not_repeat_an_article():
    for mode in ("mock", "graph"):
        p = make_predictor(mode)
        pages = [ids(p.predict("u", k=2, offset=o)) for o in (0, 2, 4)]
        assert pages == [["a1", "b1"], ["c", "d"], []]
