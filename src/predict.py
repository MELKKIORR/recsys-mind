
"""predict(user_id, k) called by the API (MLE-Ops).

mode="mock": most popular articles of the candidate pool.
mode="graph": graph retrieval on the candidate pool, with evidence paths.

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

        last = pd.concat([
            pd.read_pickle(self.cfg["processed_dir"] / f"behaviors_{s}.pkl")
            for s in ("train", "val")
        ]).sort_values("time")

        self.known_history = last.groupby("user_id")["history"].last().to_dict()

        # Store article abstracts to detect duplicate content
        self.abstract = dict(zip(
            self.rec.articles["news_id"],
            self.rec.articles["abstract"]
        ))

    def predict(self, user_id: str, k: int = 10, offset: int = 0, likes=(), dislikes=(),
                topics=None) -> list[dict]:

        history = list(self.known_history.get(user_id, [])) + [a for a in likes]

        ignored = list(self.rec.ignored.get(user_id, [])) + list(dislikes)

        banned = set(history) | set(ignored)

        pool = [a for a in self.pool if a not in banned]

        # Mock mode: rank by popularity
        if self.mode == "mock" or (not history and not topics):

            ranked = sorted(pool, key=lambda a: -self.rec.popularity.get(a, 0))

            unique_pool = []
            seen = set()

            for a in ranked:

                title = self.rec.title.get(a, "") or ""
                abstract = self.abstract.get(a, "") or ""

                key = (
                    " ".join(title.lower().split()),
                    " ".join(abstract.lower().split())
                )

                if not key[0] and not key[1]:
                    key = (a,)

                if key not in seen:
                    seen.add(key)
                    unique_pool.append(a)

            top = unique_pool[offset:offset + k]

            return [
                {
                    "article_id": a,
                    "score": float(self.rec.popularity.get(a, 0)),
                    "evidence_path": None,
                    "reason": "Popular this week"
                }
                for a in top
            ]

        # Graph mode
        profile = self.rec.build_profile(history, ignored, topics)

        scores = self.rec.score(profile, pool)

        order = sorted(range(len(pool)), key=lambda i: -scores[i])

        out = []
        seen = set()

        for i in order:

            a = pool[i]

            title = self.rec.title.get(a, "") or ""
            abstract = self.abstract.get(a, "") or ""

            key = (
                " ".join(title.lower().split()),
                " ".join(abstract.lower().split())
            )

            if not key[0] and not key[1]:
                key = (a,)

            if key in seen:
                continue

            seen.add(key)

            # Apply offset after removing duplicates
            if len(seen) <= offset:
                continue

            path = self.rec.evidence_path(profile, a)

            out.append({
                "article_id": a,
                "score": float(scores[i]),
                "evidence_path": path["entities"] if path else None,
                "reason": path["text"] if path else "Matches your favourite topics"
            })

            if len(out) >= k:
                break

        return out
