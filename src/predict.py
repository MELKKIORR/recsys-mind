"""predict(user_id, k) called by the API (MLE-Ops).

mode="mock": most popular articles of the candidate pool (fake predict used to build the site).
mode="graph": graph retrieval on the candidate pool, with evidence paths. The LLM re-ranking of the
top 10 is added in Week 3.
Already read or disliked articles, including reactions logged since the last build, are removed.
"""
import pandas as pd

from src.config import load_config
from src.graph.retrieval import GraphRecommender


class Predictor:
    def __init__(self, mode="mock", config_path="configs/default.yaml"):
        self.cfg = load_config(config_path)
        self.mode = mode
        self.rec = GraphRecommender.from_config(self.cfg)
        test = pd.read_pickle(self.cfg["processed_dir"] / "behaviors_test.pkl")
        # Demo candidate pool: articles of the test period, as if they were today's news
        self.pool = sorted({n for c in test["candidates"] for n in c})
        last = pd.concat([pd.read_pickle(self.cfg["processed_dir"] / f"behaviors_{s}.pkl")
                          for s in ("train", "val")]).sort_values("time")
        self.known_history = last.groupby("user_id")["history"].last().to_dict()

    def predict(self, user_id: str, k: int = 10, offset: int = 0, likes=(), dislikes=(),
                topics=None) -> list[dict]:
        history = list(self.known_history.get(user_id, [])) + [a for a in likes]
        ignored = list(self.rec.ignored.get(user_id, [])) + list(dislikes)
        banned = set(history) | set(dislikes)
        pool = [a for a in self.pool if a not in banned]

        if self.mode == "mock" or (not history and not topics):
            ranked = sorted(pool, key=lambda a: -self.rec.popularity.get(a, 0))
            top = ranked[offset:offset + k]
            return [{"article_id": a, "score": float(self.rec.popularity.get(a, 0)),
                     "evidence_path": None, "reason": "Popular this week"} for a in top]

        profile = self.rec.build_profile(history, ignored, topics)
        scores = self.rec.score(profile, pool)
        order = sorted(range(len(pool)), key=lambda i: -scores[i])[offset:offset + k]
        out = []
        for i in order:
            path = self.rec.evidence_path(profile, pool[i])
            out.append({"article_id": pool[i], "score": float(scores[i]),
                        "evidence_path": path["entities"] if path else None,
                        "reason": path["text"] if path else "Matches your favourite topics"})
        return out
