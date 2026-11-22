import json

from src.llm.rerank import fuse, parse_ranking

IDS = ["N1", "N2", "N3", "N4"]


def test_valid_answer():
    txt = json.dumps({"ranking": [{"id": i, "reason": "r"} for i in ["N3", "N1", "N4", "N2"]]})
    res = parse_ranking(txt, IDS)
    assert res["status"] == "ok" and res["order"] == ["N3", "N1", "N4", "N2"]


def test_unknown_and_missing_ids():
    txt = '```json\n{"ranking": [{"id": "N9"}, {"id": "N2"}, {"id": "N2"}, {"id": "N1"}]}\n```'
    res = parse_ranking(txt, IDS)
    assert res["status"] == "partial" and res["unknown"] == 1
    assert res["order"] == ["N2", "N1", "N3", "N4"]          # missing ids appended in graph order


def test_invalid_answers_fall_back_to_graph():
    assert parse_ranking("Sure! Here is my ranking", IDS)["order"] == IDS
    res = parse_ranking('{"ranking": [{"id": "N9"}, {"id": "N4"}]}', IDS)
    assert res["status"].startswith("fallback") and res["order"] == IDS


def test_fusion_weights():
    llm = ["N4", "N3", "N2", "N1"]
    assert fuse(IDS, llm, 0.0) == IDS and fuse(IDS, llm, 1.0) == llm


def test_cap_per_category():
    from src.llm.rerank import cap_per_category
    cat = {"a1": "news", "a2": "news", "a3": "news", "b1": "sports"}
    assert cap_per_category(["a1", "a2", "a3", "b1"], cat, 2, k=3) == ["a1", "a2", "b1", "a3"]
