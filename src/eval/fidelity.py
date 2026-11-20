"""Explanation fidelity (MLE-Core): does the evidence path really drive the recommendation?

For each impression, take the graph top-k. For each article that has an evidence path, remove the source
entity of that path from the reader profile, recompute the scores and check whether the article drops.
Reported: share of explained articles that drop, and their mean rank loss.
A high share means the explanation shown to the reader is the real reason of the ranking.
"""
import numpy as np


def explanation_fidelity(rec, behaviors, k=10) -> dict:
    explained, dropped, rank_loss = 0, 0, []
    for row in behaviors.itertuples(index=False):
        profile = rec.build_profile(row.history, rec.ignored.get(row.user_id, []))
        cands = list(row.candidates)
        scores = rec.score(profile, cands)
        order = list(np.argsort(-scores, kind="stable"))
        rank = {cands[i]: r for r, i in enumerate(order)}
        for i in order[:k]:
            path = rec.evidence_path(profile, cands[i])
            if not path:
                continue
            explained += 1
            new = rec.score_without_entity(profile, cands, path["source_entity"])
            new_order = list(np.argsort(-new, kind="stable"))
            new_rank = new_order.index(i)
            if new_rank > rank[cands[i]]:
                dropped += 1
                rank_loss.append(new_rank - rank[cands[i]])
    return {"explained_in_top_k": explained,
            "share_explained_that_drop": round(dropped / max(1, explained), 4),
            "mean_rank_loss_when_drop": round(float(np.mean(rank_loss)), 2) if rank_loss else 0.0}
