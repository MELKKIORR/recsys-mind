"""predict(user_id, k) called by the API (MLE-Ops).

mode="mock": most popular articles of the candidate pool (fake predict used to build the site).
mode="graph": graph retrieval on the candidate pool, with evidence paths, then LLM re-ranking of the
top 10 (from the disk cache, or the graph ranking if the LLM answer is invalid or the call fails).
Already read or disliked articles, including reactions logged since the last build, are removed.
"""
import pandas as pd

from src.config import load_config
from src.graph.retrieval import GraphRecommender
from src.llm.client import LLMClient
from src.llm.profile_summary import summarize
from src.llm.rerank import Reranker, cap_per_category, explanation


class Predictor:
    def __init__(self, mode="mock", config_path="configs/default.yaml"):
        self.cfg = load_config(config_path)
        self.mode = mode
        self.rec = GraphRecommender.from_config(self.cfg)
        self.llm = LLMClient.from_config(self.cfg)
        self.reranker = Reranker.from_config(self.cfg, self.rec, self.llm)
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
        # Only the best graph candidates go to the LLM: re-rank the top N, keep the rest in graph order
        shortlist_idx = sorted(range(len(pool)), key=lambda i: -scores[i])[:max(k + offset, 50)]
        shortlist = [pool[i] for i in shortlist_idx]
        res = self.reranker.rerank(profile, shortlist, [scores[i] for i in shortlist_idx])
        order = cap_per_category(res["order"], self.rec.category,
                                 self.cfg["rerank"].get("max_per_category", 0), k=offset + k)
        out = []
        for a in order[offset:offset + k]:
            path = res["paths"].get(a) or self.rec.evidence_path(profile, a)
            out.append({"article_id": a, "score": float(scores[pool.index(a)]),
                        "evidence_path": path["entities"] if path else None,
                        "reason": explanation(path, res["reasons"].get(a)),
                        "llm_status": res["status"]})
        return out

    def profile_summary(self, user_id: str, likes=(), dislikes=(), topics=None) -> dict:
        """Two-sentence LLM summary of the reader for the Profile page (cached, template if it fails)."""
        history = list(self.known_history.get(user_id, [])) + list(likes)
        ignored = list(self.rec.ignored.get(user_id, [])) + list(dislikes)
        profile = self.rec.build_profile(history, ignored, topics)
        return summarize(self.rec, profile, self.llm)
