"""Multi-hop graph retrieval with evidence paths (MLE-Core).

Forward pass for one impression:
  1. structured profile from the history (entity and category weights, recent clicks count more)
     and from the ignored articles (negative signal)
  2. personalised PageRank on the entity graph, truncated to `hops` hops
  3. score of each candidate = entity relevance + category affinity - ignored penalty
  4. evidence path for each candidate: history article -> entity -> entity -> candidate
A candidate that nobody clicked is scored through its entities (cold start).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def latest_artefacts(artefacts_dir: Path) -> Path:
    name = (Path(artefacts_dir) / "LATEST").read_text().strip()
    return Path(artefacts_dir) / name


@dataclass
class Profile:
    entity_w: np.ndarray                 # (E,) sums to 1 when the history has entities
    category_w: dict                     # category -> weight, sums to 1
    ignored_entity_w: np.ndarray         # (E,)
    ignored_category_w: dict
    diversity: float                     # normalised entropy of the clicked categories, 0 to 1
    history: list = field(default_factory=list)

    def top_entities(self, labels, k=10):
        idx = np.argsort(-self.entity_w)[:k]
        return [(labels[i], round(float(self.entity_w[i]), 4)) for i in idx if self.entity_w[i] > 0]


class GraphRecommender:
    def __init__(self, artefact_dir: Path, hops=3, damping=0.85, recency_decay=0.9,
                 ignored_weight=0.5, category_weight=0.3, entity_idf=True):
        d = Path(artefact_dir)
        self.ae = sp.load_npz(d / "article_entity.npz").tocsr()
        ee = sp.load_npz(d / "entity_entity.npz").tocsr()
        self.articles = pd.read_pickle(d / "articles.pkl")
        self.entities = pd.read_pickle(d / "entities.pkl")
        stats = pd.read_pickle(d / "train_stats.pkl")
        self.popularity, self.ignored = stats["popularity"], stats["ignored"]
        self.art_idx = {n: i for i, n in enumerate(self.articles["news_id"])}
        self.category = dict(zip(self.articles["news_id"], self.articles["category"]))
        self.title = dict(zip(self.articles["news_id"], self.articles["title"]))
        self.ent_labels = self.entities["label"].tolist()
        self.ee = ee
        deg = np.asarray(ee.sum(axis=1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=np.float64), where=deg > 0)
        self.p_t = (sp.diags(inv) @ ee).T.tocsr()          # transpose of the row-normalised matrix
        self.n_ent_per_article = np.asarray(self.ae.sum(axis=1)).ravel()
        # Hub entities ("Associated Press", "United States") appear in thousands of articles and link
        # everything to everything. IDF weighting lowers them in the profile, the scores and the paths.
        df = np.asarray(self.ae.sum(axis=0)).ravel()
        idf = np.log((1 + self.ae.shape[0]) / (1 + df))
        self.idf = idf / idf.max() if entity_idf and idf.max() > 0 else np.ones_like(idf)
        self.hops, self.damping, self.recency_decay = hops, damping, recency_decay
        self.ignored_weight, self.category_weight = ignored_weight, category_weight

    @classmethod
    def from_config(cls, cfg: dict, artefact_dir: Path | None = None):
        r = cfg["retrieval"]
        return cls(artefact_dir or latest_artefacts(cfg["artefacts_dir"]), hops=r["hops"],
                   damping=r["damping"], recency_decay=r["recency_decay"],
                   ignored_weight=r["ignored_weight"], category_weight=r["category_weight"],
                   entity_idf=r.get("entity_idf", True))

    # 1. profile -------------------------------------------------------------
    def _weights(self, articles, weights):
        ent = np.zeros(self.ae.shape[1])
        cats = {}
        for a, w in zip(articles, weights):
            i = self.art_idx.get(a)
            if i is None:
                continue
            row = self.ae.getrow(i)
            ent[row.indices] += w * self.idf[row.indices]
            c = self.category[a]
            cats[c] = cats.get(c, 0.0) + w
        if ent.sum() > 0:
            ent /= ent.sum()
        total = sum(cats.values())
        cats = {c: v / total for c, v in cats.items()} if total else {}
        return ent, cats

    def build_profile(self, history: list, ignored: list | None = None,
                      topics: list | None = None) -> Profile:
        n = len(history)
        w = [self.recency_decay ** (n - 1 - i) for i in range(n)]   # MIND history is oldest first
        ent, cats = self._weights(history, w)
        if topics and not history:                                   # onboarding: 3 chosen topics
            # each chosen topic gets the same weight, so one busy topic cannot fill the whole profile
            ent = np.zeros(self.ae.shape[1])
            for t in topics:
                pool = sorted((a for a in self.popularity if self.category.get(a) == t),
                              key=lambda a: -self.popularity[a])[:20]
                e, _ = self._weights(pool, [1.0] * len(pool))
                ent += e / len(topics)
            cats = {t: 1.0 / len(topics) for t in topics}
        ign = ignored or []
        ign_ent, ign_cats = self._weights(ign, [1.0] * len(ign))
        if cats:
            p = np.array(list(cats.values()))
            div = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        else:
            div = 0.0
        return Profile(ent, cats, ign_ent, ign_cats, div, list(history))

    # 2. multi-hop retrieval -------------------------------------------------
    def personalised_pagerank(self, seed: np.ndarray) -> np.ndarray:
        """r = (1-d) * sum_{k=0..hops} d^k * seed P^k (truncated PPR)."""
        d = self.damping
        x = seed.astype(np.float64)
        r = (1 - d) * x
        for _ in range(self.hops):
            x = d * (self.p_t @ x)
            r += (1 - d) * x
        return r

    # 3. scoring -------------------------------------------------------------
    def score(self, profile: Profile, candidates: list) -> np.ndarray:
        r = self.personalised_pagerank(profile.entity_w)
        idx = np.array([self.art_idx.get(c, -1) for c in candidates])
        known = idx >= 0
        scores = np.zeros(len(candidates))
        if not known.any():
            return scores
        sub = self.ae[idx[known]]
        n = np.maximum(1.0, self.n_ent_per_article[idx[known]])
        ent_score = (sub @ (r * self.idf)) / n
        ign_score = (sub @ profile.ignored_entity_w) / n
        cats = [self.category[c] for c, k in zip(candidates, known) if k]
        cat_aff = np.array([profile.category_w.get(c, 0.0) for c in cats])
        cat_ign = np.array([profile.ignored_category_w.get(c, 0.0) for c in cats])
        scores[known] = (ent_score + self.category_weight * cat_aff
                         - self.ignored_weight * (ign_score + cat_ign))
        return scores

    # 4. evidence path -------------------------------------------------------
    def evidence_path(self, profile: Profile, candidate: str) -> dict | None:
        i = self.art_idx.get(candidate)
        if i is None or profile.entity_w.sum() == 0:
            return None
        cand_ents = self.ae.getrow(i).indices
        if len(cand_ents) == 0:
            return None
        prof_ents = np.nonzero(profile.entity_w)[0]
        best, best_val = None, 0.0
        for ec in cand_ents:
            # Paths use idf squared: an explanation must name a specific subject, not a hub
            if profile.entity_w[ec] * self.idf[ec] ** 2 > best_val:   # 0 hop: shared entity
                best, best_val = (ec, ec), profile.entity_w[ec] * self.idf[ec] ** 2
            links = self.ee.getrow(ec)                                # 1 hop: linked entity
            common = np.intersect1d(links.indices, prof_ents)
            for eu in common:
                v = (profile.entity_w[eu] * (self.idf[eu] * self.idf[ec]) ** 2
                     * self.ee[ec, eu] / (1 + self.ee[ec, eu]))
                if v > best_val:
                    best, best_val = (eu, ec), v
        if best is None:
            return None
        eu, ec = best
        source = next((a for a in reversed(profile.history)
                       if a in self.art_idx and eu in self.ae.getrow(self.art_idx[a]).indices), None)
        labels = [self.ent_labels[eu]] if eu == ec else [self.ent_labels[eu], self.ent_labels[ec]]
        return {"source_article": source, "source_title": self.title.get(source, ""),
                "entities": labels, "source_entity": int(eu),
                "text": (f"Because you read: {self.title[source]} ({' -> '.join(labels)})" if source
                         else f"Linked to your topics: {' -> '.join(labels)}")}

    def score_without_entity(self, profile: Profile, candidates: list, entity: int) -> np.ndarray:
        """Scores after removing one entity from the profile (used to test explanation fidelity)."""
        w = profile.entity_w.copy()
        w[entity] = 0.0
        if w.sum() > 0:
            w /= w.sum()
        cut = Profile(w, profile.category_w, profile.ignored_entity_w, profile.ignored_category_w,
                      profile.diversity, profile.history)
        return self.score(cut, candidates)

    def forward(self, history, candidates, ignored=None, topics=None, with_paths=True):
        profile = self.build_profile(history, ignored, topics)
        scores = self.score(profile, candidates)
        paths = [self.evidence_path(profile, c) for c in candidates] if with_paths else None
        return scores, paths, profile
