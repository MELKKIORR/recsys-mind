"""Qualitative sanity check for the Week 3 report (MLE-Core): 3 sample readers and their top 10.

  python scripts/sample_users.py [config]       -> prints and writes results/sample_users.md

Reader 1: long history. Reader 2: fewer than 5 clicks. Reader 3: new reader who picks 3 topics.
For readers 1 and 2 we use one real test impression, so we can show which recommended articles the
reader really clicked. For each top 10 we report the number of categories and the mean popularity,
next to the most-popular ranking of the same candidates.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config  # noqa: E402
from src.graph.retrieval import GraphRecommender  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.llm.rerank import Reranker, cap_per_category, explanation  # noqa: E402


def top10_block(rec, title, history, cands, labels, res, max_per_category=0):
    res = {**res, "order": cap_per_category(res["order"], rec.category, max_per_category)}
    lines = [f"## {title}", ""]
    hist_cats = pd.Series([rec.category.get(a) for a in history]).value_counts().head(4)
    lines.append(f"History: {len(history)} clicks, mostly " +
                 ", ".join(f"{c} ({n})" for c, n in hist_cats.items()) if history else "History: none")
    for a in history[-3:]:
        lines.append(f"- read: {rec.title.get(a, a)} ({rec.category.get(a)})")
    lines += ["", "| # | Article | Category | Clicked | Why |", "| --- | --- | --- | --- | --- |"]
    lab = dict(zip(cands, labels)) if labels is not None else {}
    for r, a in enumerate(res["order"][:10], 1):
        why = explanation(res["paths"].get(a), res["reasons"].get(a)).replace("|", "/")
        clicked = {1: "yes", 0: "no"}.get(lab.get(a), "-")
        lines.append(f"| {r} | {rec.title.get(a, a)[:70]} | {rec.category.get(a)} | {clicked} | {why[:110]} |")
    top = res["order"][:10]
    pop_top = sorted(cands, key=lambda a: -rec.popularity.get(a, 0))[:10]
    lines += ["", f"Our top 10: {len({rec.category.get(a) for a in top})} categories, mean popularity "
              f"{np.mean([rec.popularity.get(a, 0) for a in top]):.1f} clicks. Most-popular top 10: "
              f"{len({rec.category.get(a) for a in pop_top})} categories, mean popularity "
              f"{np.mean([rec.popularity.get(a, 0) for a in pop_top]):.1f}. "
              f"Articles never seen in training in our top 10: "
              f"{sum(rec.popularity.get(a, 0) == 0 for a in top)}. LLM status: {res['status']}.", ""]
    return lines


def main(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    rec = GraphRecommender.from_config(cfg)
    reranker = Reranker.from_config(cfg, rec, LLMClient.from_config(cfg))
    cap = cfg["rerank"].get("max_per_category", 0)        # same display rule as the website
    test = pd.read_pickle(cfg["processed_dir"] / "behaviors_test.pkl")
    test = test[test["labels"].map(sum) > 0]
    n_hist = test["history"].map(len)
    rng = np.random.default_rng(cfg["eval"]["seed"])
    many = test["candidates"].map(len) >= min(15, test["candidates"].map(len).max())
    heavy = test[(n_hist >= 20) & many]
    if heavy.empty:                                   # small datasets: take the longest histories
        heavy = test[many & (n_hist >= n_hist.quantile(0.9))]
    light = test[(n_hist >= 1) & (n_hist < 5) & many]
    out = ["# Qualitative sanity check: 3 sample readers", ""]
    for title, pool in (("Reader 1: long history", heavy), ("Reader 2: fewer than 5 clicks", light)):
        if pool.empty:
            continue
        row = pool.iloc[rng.integers(len(pool))]
        profile = rec.build_profile(row.history, rec.ignored.get(row.user_id, []))
        cands = list(row.candidates)
        res = reranker.rerank(profile, cands, rec.score(profile, cands))
        out += top10_block(rec, f"{title} ({row.user_id}, impression {row.imp_id})",
                           row.history, cands, row.labels, res, cap)
    cats = pd.Series([rec.category.get(a) for a in rec.popularity]).value_counts().index[:3].tolist()
    pool = sorted({a for c in test["candidates"] for a in c})
    profile = rec.build_profile([], [], topics=cats)
    scores = rec.score(profile, pool)
    short = [pool[i] for i in np.argsort(-scores)[:50]]
    res = reranker.rerank(profile, short, sorted(scores, reverse=True)[:50])
    out += top10_block(rec, f"Reader 3: new reader who picked {', '.join(cats)}", [], short, None, res, cap)
    text = "\n".join(out)
    print(text)
    path = ROOT / "results" / "sample_users.md"
    path.write_text(text, encoding="utf-8")
    print("Saved", path)


if __name__ == "__main__":
    main(*sys.argv[1:])
