"""Impression-level ranking metrics (standard MIND metrics + Recall@k)."""
import numpy as np


def auc(labels, scores) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def _order(scores):
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def mrr(labels, scores) -> float:
    y = np.asarray(labels)[_order(scores)]
    rr = y / (np.arange(len(y)) + 1)
    return float(rr.sum() / max(1, y.sum()))


def dcg(y, k):
    y = y[:k]
    return float(((2 ** y - 1) / np.log2(np.arange(len(y)) + 2)).sum())


def ndcg(labels, scores, k) -> float:
    labels = np.asarray(labels)
    ideal = dcg(np.sort(labels)[::-1], k)
    return dcg(labels[_order(scores)], k) / ideal if ideal > 0 else float("nan")


def recall(labels, scores, k) -> float:
    labels = np.asarray(labels)
    if labels.sum() == 0:
        return float("nan")
    return float(labels[_order(scores)][:k].sum() / labels.sum())


def impression_metrics(labels, scores, k_values=(5, 10), recall_k=20) -> dict:
    out = {"auc": auc(labels, scores), "mrr": mrr(labels, scores)}
    for k in k_values:
        out[f"ndcg@{k}"] = ndcg(labels, scores, k)
    out[f"recall@{recall_k}"] = recall(labels, scores, recall_k)
    return out
