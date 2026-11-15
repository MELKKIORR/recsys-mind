import math

from src.eval.metrics import auc, mrr, ndcg, recall


def test_perfect_ranking():
    y, s = [1, 0, 0], [3, 2, 1]
    assert auc(y, s) == 1.0 and mrr(y, s) == 1.0 and ndcg(y, s, 5) == 1.0 and recall(y, s, 1) == 1.0


def test_worst_ranking():
    y, s = [1, 0, 0], [1, 2, 3]
    assert auc(y, s) == 0.0
    assert math.isclose(mrr(y, s), 1 / 3)
    assert math.isclose(ndcg(y, s, 5), 1 / math.log2(4))


def test_ties_give_half_auc():
    assert auc([1, 0], [1, 1]) == 0.5
