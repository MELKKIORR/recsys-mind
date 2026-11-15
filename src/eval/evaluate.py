"""Evaluation harness (PL): one script for every model, baselines included.

    python -m src.eval.evaluate --split val --models random popular graph --sample 2000 --seeds 1 2 3

A model is any function score(row) -> list of scores, one per candidate of the impression.
Results (mean of the seeds) are printed and saved in results/.
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.eval.metrics import impression_metrics


def random_model(seed):
    rng = np.random.default_rng(seed)
    return lambda row: rng.random(len(row.candidates))


def popular_model(popularity):
    return lambda row: [popularity.get(c, 0) for c in row.candidates]


def graph_model(rec):
    def score(row):
        ignored = rec.ignored.get(row.user_id, [])
        return rec.forward(row.history, row.candidates, ignored=ignored, with_paths=False)[0]
    return score


def evaluate(behaviors: pd.DataFrame, score_fn, k_values=(5, 10), recall_k=20,
             cold_articles: set | None = None) -> dict:
    rows = []
    for row in behaviors.itertuples(index=False):
        m = impression_metrics(row.labels, score_fn(row), k_values, recall_k)
        m["cold"] = bool(cold_articles) and any(c in cold_articles for c in row.candidates)
        m["short_history"] = len(row.history) < 5
        rows.append(m)
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("cold", "short_history")]
    res = {"all": df[metric_cols].mean(skipna=True).round(4).to_dict(), "n": len(df)}
    for slice_name in ("cold", "short_history"):
        sub = df[df[slice_name]]
        if len(sub):
            res[slice_name] = sub[metric_cols].mean(skipna=True).round(4).to_dict()
            res[slice_name]["n"] = len(sub)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", nargs="+", default=["random", "popular", "graph"])
    ap.add_argument("--sample", type=int, default=0, help="0 = all impressions")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = cfg["processed_dir"]
    beh = pd.read_pickle(proc / f"behaviors_{args.split}.pkl")
    train = pd.read_pickle(proc / "behaviors_train.pkl")
    seen = {n for h, c in zip(train["history"], train["candidates"]) for n in (*h, *c)}
    cold = {n for c in beh["candidates"] for n in c} - seen
    k_values, recall_k = cfg["eval"]["k_values"], cfg["eval"]["recall_k"]

    rec = None
    if "graph" in args.models or "popular" in args.models:
        from src.graph.retrieval import GraphRecommender
        rec = GraphRecommender.from_config(cfg)

    results = {}
    for name in args.models:
        runs = []
        for seed in args.seeds:
            sample = beh.sample(n=min(args.sample, len(beh)), random_state=seed) if args.sample else beh
            fn = {"random": lambda: random_model(seed),
                  "popular": lambda: popular_model(rec.popularity),
                  "graph": lambda: graph_model(rec)}[name]()
            t0 = time.time()
            r = evaluate(sample, fn, k_values, recall_k, cold)
            r["seconds"] = round(time.time() - t0, 1)
            runs.append(r)
        mean = pd.DataFrame([r["all"] for r in runs]).mean().round(4).to_dict()
        results[name] = {"mean_over_seeds": mean, "runs": runs}
        print(f"{name:8s} " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))

    out = ROOT / "results" / f"eval_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"split": args.split, "sample": args.sample, "seeds": args.seeds,
                   "config": {k: v for k, v in cfg.items() if isinstance(v, (dict, int, float))},
                   "results": results}, f, indent=2, default=str)
    print("Saved", out)


if __name__ == "__main__":
    main()
