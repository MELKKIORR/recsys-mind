"""LLM re-ranking of the top candidates, bound to the given identifiers (MLE-Core).

The LLM never answers freely: it gets the structured profile and the top N graph candidates (id, title,
category, evidence path) and must return JSON with a ranking of those ids and a one-line reason each.
The code then checks the answer:
  - ids that were not given are dropped (counted as unknown ids)
  - duplicates are dropped, missing ids are appended in graph order
  - if the JSON is invalid or too few ids are valid, the graph ranking is used (fallback)
The final order mixes the graph rank and the LLM rank: fusion_weight 0 = graph only, 1 = LLM only.
"""
import json
import re

import numpy as np

SYSTEM = {
    "v1": (
        "You re-rank news articles for one reader of a news site. Use only the article ids you are given. "
        "Order them from most to least relevant for this reader, using the reader profile and the evidence "
        "path of each article (how it is linked to articles the reader opened). Push down articles from "
        "categories or about entities the reader ignored. "
        'Answer with JSON only, no other text: {"ranking": [{"id": "<id>", "reason": "<one short sentence>"}]} '
        "with every given id exactly once, best first."),
    "v2": (
        "You help a news site explain and order recommendations. Use only the given article ids. "
        "Put first the articles whose evidence path links them strongly to what the reader read, keep some "
        "variety of categories in the first five, and push down what looks like the reader's ignored topics. "
        "Each reason must name the link in plain words (for example: you follow Tesla, Panasonic supplies it). "
        'Answer with JSON only: {"ranking": [{"id": "<id>", "reason": "<one short sentence>"}]}, every id once.'),
}

SYSTEM_LLM_ONLY = (
    "You recommend news articles to a reader from the titles of articles they read before. Use only the "
    'given ids. Answer with JSON only: {"ranking": [{"id": "<id>", "reason": "<one short sentence>"}]}, '
    "every id once, best first.")


def describe_profile(rec, profile, k=8) -> dict:
    """Structured profile sent to the LLM (the ranking itself uses the numbers, not this text)."""
    labels = rec.ent_labels
    ign_ents = np.argsort(-profile.ignored_entity_w)[:5]
    return {
        "top_categories": {c: round(w, 2) for c, w in
                           sorted(profile.category_w.items(), key=lambda x: -x[1])[:5]},
        "top_entities": [l for l, _ in profile.top_entities(labels, k)],
        "ignored_categories": [c for c, _ in
                               sorted(profile.ignored_category_w.items(), key=lambda x: -x[1])[:3]],
        "ignored_entities": [labels[i] for i in ign_ents if profile.ignored_entity_w[i] > 0],
        "diversity": round(profile.diversity, 2),
    }


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def parse_ranking(text: str, given_ids: list, min_valid_share=0.5) -> dict:
    """Check the LLM answer against the given ids. Returns order, reasons, status, unknown ids."""
    data = extract_json(text)
    items = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {"order": list(given_ids), "reasons": {}, "status": "invalid_json", "unknown": 0}
    allowed, order, reasons, unknown = set(given_ids), [], {}, 0
    for it in items:
        i = it.get("id") if isinstance(it, dict) else it
        if i not in allowed:
            unknown += 1
            continue
        if i in reasons or i in order:
            continue
        order.append(i)
        if isinstance(it, dict) and isinstance(it.get("reason"), str):
            reasons[i] = it["reason"].strip()[:200]
    if len(order) < min_valid_share * len(given_ids):
        return {"order": list(given_ids), "reasons": {}, "status": "fallback_too_few_ids",
                "unknown": unknown}
    status = "ok" if len(order) == len(given_ids) and unknown == 0 else "partial"
    order += [i for i in given_ids if i not in order]       # missing ids keep their graph order
    return {"order": order, "reasons": reasons, "status": status, "unknown": unknown}


def fuse(graph_order: list, llm_order: list, weight: float) -> list:
    """Mix two rankings of the same ids: score = (1-w) * graph position + w * LLM position."""
    n = len(graph_order)
    if n <= 1:
        return list(graph_order)
    pos_g = {a: 1 - r / (n - 1) for r, a in enumerate(graph_order)}
    pos_l = {a: 1 - r / (n - 1) for r, a in enumerate(llm_order)}
    return sorted(graph_order, key=lambda a: -((1 - weight) * pos_g[a] + weight * pos_l[a]))


class Reranker:
    def __init__(self, rec, client, top_n=10, fusion_weight=0.5, min_valid_share=0.5,
                 prompt_version="v1"):
        self.rec, self.client = rec, client
        self.top_n, self.weight, self.min_share = top_n, fusion_weight, min_valid_share
        self.prompt_version = prompt_version
        self.stats = {"impressions": 0, "fallback": 0, "partial": 0, "unknown_ids": 0, "llm_errors": 0}

    @classmethod
    def from_config(cls, cfg, rec, client):
        r = cfg["rerank"]
        return cls(rec, client, r["top_n"], r["fusion_weight"], r["min_valid_share"],
                   cfg["llm"]["prompt_version"])

    def _ask(self, system, payload, ids):
        try:
            text = self.client.complete(system, json.dumps(payload, ensure_ascii=False))
        except RuntimeError:
            self.stats["llm_errors"] += 1
            return {"order": list(ids), "reasons": {}, "status": "fallback_llm_error", "unknown": 0}
        return parse_ranking(text, ids, self.min_share)

    def _record(self, res):
        self.stats["impressions"] += 1
        self.stats["unknown_ids"] += res["unknown"]
        if res["status"].startswith("fallback") or res["status"] == "invalid_json":
            self.stats["fallback"] += 1
        elif res["status"] == "partial":
            self.stats["partial"] += 1

    def rerank(self, profile, candidates: list, graph_scores, paths=None) -> dict:
        """Re-rank the top N graph candidates. Returns final scores aligned with `candidates`."""
        graph_scores = np.asarray(graph_scores, dtype=float)
        order = list(np.argsort(-graph_scores, kind="stable"))
        top = [candidates[i] for i in order[:self.top_n]]
        rest = [candidates[i] for i in order[self.top_n:]]
        if paths is None:
            paths = [self.rec.evidence_path(profile, c) for c in candidates]
        path_of = dict(zip(candidates, paths))
        payload = {"reader": describe_profile(self.rec, profile), "candidates": [
            {"id": c, "title": self.rec.title.get(c, ""), "category": self.rec.category.get(c, ""),
             "evidence": " -> ".join(path_of[c]["entities"]) if path_of.get(c) else "none"}
            for c in top]}
        res = self._ask(SYSTEM[self.prompt_version], payload, top)
        self._record(res)
        final = fuse(top, res["order"], self.weight) + rest
        n = len(final)
        rank = {a: r for r, a in enumerate(final)}
        scores = np.array([float(n - rank[c]) for c in candidates])
        return {"order": final, "scores": scores, "reasons": res["reasons"], "status": res["status"],
                "paths": path_of}

    def llm_only(self, history: list, candidates: list) -> np.ndarray:
        """Ablation: the LLM ranks the candidates from the history titles alone (no graph)."""
        payload = {"history": [self.rec.title.get(a, "") for a in history[-10:]],
                   "candidates": [{"id": c, "title": self.rec.title.get(c, ""),
                                   "category": self.rec.category.get(c, "")} for c in candidates]}
        res = self._ask(SYSTEM_LLM_ONLY, payload, list(candidates))
        self._record(res)
        n = len(res["order"])
        rank = {a: r for r, a in enumerate(res["order"])}
        return np.array([float(n - rank[c]) for c in candidates])

    def fallback_rate(self) -> float:
        return self.stats["fallback"] / max(1, self.stats["impressions"])


def cap_per_category(order: list, category: dict, max_per_category: int, k: int = 10) -> list:
    """Display rule for the website: at most `max_per_category` articles of one category in the top k.
    Articles pushed out keep their order after the top k. Not used in the offline evaluation."""
    if not max_per_category:
        return list(order)
    top, rest, count = [], [], {}
    for a in order:
        c = category.get(a)
        if len(top) < k and count.get(c, 0) < max_per_category:
            top.append(a)
            count[c] = count.get(c, 0) + 1
        else:
            rest.append(a)
    return top + rest


def explanation(path: dict | None, reason: str | None) -> str:
    """Text shown under a recommendation: evidence path first, LLM reason second."""
    parts = []
    if path:
        parts.append(path["text"])
    if reason:
        parts.append(reason)
    return " | ".join(parts) if parts else "Matches your favourite topics"
