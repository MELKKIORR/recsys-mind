"""predict(user_id, k) called by the API (MLE-Ops, serving part SE).

mode="mock": most popular articles of the candidate pool (fake predict used to build the site).
mode="graph": graph retrieval on the candidate pool, with evidence paths, then LLM re-ranking of the
top 10 (from the disk cache, or the graph ranking if the LLM answer is invalid or the call fails).
Already read or disliked articles, including reactions logged since the last build, are removed.

Serving (SE): the website must answer in under 2 s, so by default the API reads the LLM answer from the
disk cache only (live_llm=False). On a cache miss the graph ranking is served (llm_status
"fallback_cache_miss") and the API calls predict again with live_llm=True in the background, which
fills the cache: the next request of that reader gets the LLM ranking.
"""
import pandas as pd

from src.config import load_config
from src.graph.retrieval import GraphRecommender
from src.llm.client import LLMClient
from src.llm.rerank import Reranker, cap_per_category, explanation


class Predictor:
    def __init__(self, mode="mock", config_path="configs/default.yaml", cache_only=True):
        self.cfg = load_config(config_path)
        self.mode = mode
        self.rec = GraphRecommender.from_config(self.cfg)
        # Same cache directory: the live reranker writes the answers the cache-only one reads
        self.live_reranker = Reranker.from_config(self.cfg, self.rec, LLMClient.from_config(self.cfg))
        self.reranker = (Reranker.from_config(self.cfg, self.rec,
                                              LLMClient.from_config(self.cfg, cache_only=True))
                         if cache_only else self.live_reranker)
        test = pd.read_pickle(self.cfg["processed_dir"] / "behaviors_test.pkl")
        # Demo candidate pool: articles of the test period, as if they were today's news
        self.pool = sorted({n for c in test["candidates"] for n in c})
        last = pd.concat([pd.read_pickle(self.cfg["processed_dir"] / f"behaviors_{s}.pkl")
                          for s in ("train", "val")]).sort_values("time")
        self.known_history = last.groupby("user_id")["history"].last().to_dict()

    def predict(self, user_id: str, k: int = 10, offset: int = 0, likes=(), dislikes=(),
                topics=None, live_llm=False) -> list[dict]:
        history = list(self.known_history.get(user_id, [])) + [a for a in likes]
        ignored = list(self.rec.ignored.get(user_id, [])) + list(dislikes)
        banned = set(history) | set(dislikes)
        pool = [a for a in self.pool if a not in banned]

        if self.mode == "mock" or (not history and not topics):
            ranked = sorted(pool, key=lambda a: -self.rec.popularity.get(a, 0))
            top = ranked[offset:offset + k]
            return [{"article_id": a, "score": float(self.rec.popularity.get(a, 0)),
                     "evidence_path": None, "source_title": None, "reason": "Popular this week",
                     "llm_reason": None, "llm_status": "not_used"} for a in top]

        profile = self.rec.build_profile(history, ignored, topics)
        scores = self.rec.score(profile, pool)
        score_of = dict(zip(pool, scores))
        # Only the best graph candidates go to the LLM: re-rank the top N, keep the rest in graph order
        shortlist_idx = sorted(range(len(pool)), key=lambda i: -scores[i])[:max(k + offset, 50)]
        shortlist = [pool[i] for i in shortlist_idx]
        reranker = self.live_reranker if live_llm else self.reranker
        res = reranker.rerank(profile, shortlist, [scores[i] for i in shortlist_idx])
        order = cap_per_category(res["order"], self.rec.category,
                                 self.cfg["rerank"].get("max_per_category", 0), k=offset + k)
        out = []
        for a in order[offset:offset + k]:
            path = res["paths"].get(a) or self.rec.evidence_path(profile, a)
            out.append({"article_id": a, "score": float(score_of[a]),
                        "evidence_path": path["entities"] if path else None,
                        "source_title": path["source_title"] or None if path else None,
                        "reason": explanation(path, res["reasons"].get(a)),
                        "llm_reason": res["reasons"].get(a) or None,
                        "llm_status": res["status"]})
        return out
