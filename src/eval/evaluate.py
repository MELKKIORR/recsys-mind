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
from src.eval.metrics import impression_metrics, ndcg


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


def graph_variant(rec, **changes):
    """Same graph model with one component changed (ablations)."""
    import copy
    r = copy.copy(rec)
    for k, v in changes.items():
        setattr(r, k, v)
    return graph_model(r)


def full_model(rec, reranker):
    """Graph retrieval + LLM re-ranking of the top N (the full system)."""
    def score(row):
        profile = rec.build_profile(row.history, rec.ignored.get(row.user_id, []))
        cands = list(row.candidates)
        g = rec.score(profile, cands)
        return reranker.rerank(profile, cands, g)["scores"]
    return score


def llm_only_model(reranker):
    return lambda row: reranker.llm_only(list(row.history), list(row.candidates))


def evaluate(behaviors: pd.DataFrame, score_fn, k_values=(5, 10), recall_k=20,
             cold_articles: set | None = None, popularity: dict | None = None,
             category: dict | None = None) -> dict:
    rows = []
    for row in behaviors.itertuples(index=False):
        scores = score_fn(row)
        m = impression_metrics(row.labels, scores, k_values, recall_k)
        top = [row.candidates[i] for i in np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:10]]
        if popularity is not None:
            m["top10_mean_popularity"] = float(np.mean([popularity.get(a, 0) for a in top]))
        if category is not None:
            m["top10_n_categories"] = len({category.get(a) for a in top})
        if cold_articles:
            # Cold-start NDCG: rank only the articles never seen in training, among themselves
            mask = np.array([c in cold_articles for c in row.candidates])
            lab = np.asarray(row.labels)[mask]
            m["cold_ndcg@10"] = (ndcg(lab, np.asarray(scores, dtype=float)[mask], 10)
                                 if mask.sum() >= 2 and lab.sum() > 0 else float("nan"))
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


TEST_LOG = ROOT / "results" / "final_test_log.json"


def guard_test_split(args) -> None:
    """The test split is touched once, after tuning on validation. Checked before the run."""
    if not args.final:
        raise SystemExit("The test split is only for the final evaluation. Tune on --split val, "
                         "then run once with --split test --final.")
    log = json.loads(TEST_LOG.read_text()) if TEST_LOG.exists() else []
    if any(e.get("status") == "completed" for e in log) and not args.force:
        done = [e for e in log if e.get("status") == "completed"][-1]
        raise SystemExit(f"The test split was already used on {done['date']} (see {TEST_LOG}). "
                         "Add --force only if you have a real reason, it will be logged.")


def record_test_run(args, results_file) -> None:
    """Logged only after the evaluation finished and its results were saved."""
    from src.graph.build_graph import git_commit
    log = json.loads(TEST_LOG.read_text()) if TEST_LOG.exists() else []
    forced = any(e.get("status") == "completed" for e in log)
    log.append({"date": f"{datetime.now():%Y-%m-%d %H:%M}", "commit": git_commit(), "status": "completed",
                "models": args.models, "sample": args.sample, "seeds": args.seeds,
                "results": str(results_file.name), "forced": forced})
    TEST_LOG.parent.mkdir(exist_ok=True)
    TEST_LOG.write_text(json.dumps(log, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", nargs="+", default=["random", "popular", "graph"],
                    help="random popular graph graph_no_ignored graph_1hop llm_only full")
    ap.add_argument("--sample", type=int, default=0, help="0 = all impressions")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--final", action="store_true",
                    help="required for --split test: the test split is used once, for the final systems")
    ap.add_argument("--force", action="store_true", help="run the test split again (logged)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.split == "test":
        guard_test_split(args)
    proc = cfg["processed_dir"]
    beh = pd.read_pickle(proc / f"behaviors_{args.split}.pkl")
    train = pd.read_pickle(proc / "behaviors_train.pkl")
    seen = {n for h, c in zip(train["history"], train["candidates"]) for n in (*h, *c)}
    cold = {n for c in beh["candidates"] for n in c} - seen
    k_values, recall_k = cfg["eval"]["k_values"], cfg["eval"]["recall_k"]

    from src.graph.retrieval import GraphRecommender
    rec = GraphRecommender.from_config(cfg)
    reranker = None
    if {"full", "llm_only"} & set(args.models):
        try:                                 # LLM re-ranking module (MLE-Core), added in its own PR
            from src.llm.client import LLMClient
            from src.llm.rerank import Reranker
        except ImportError:
            raise SystemExit("Models full and llm_only need the LLM module (src/llm/rerank.py).")
        reranker = Reranker.from_config(cfg, rec, LLMClient.from_config(cfg))

    results = {}
    for name in args.models:
        runs = []
        for seed in args.seeds:
            sample = beh.sample(n=min(args.sample, len(beh)), random_state=seed) if args.sample else beh
            fn = {"random": lambda: random_model(seed),
                  "popular": lambda: popular_model(rec.popularity),
                  "graph": lambda: graph_model(rec),
                  "graph_no_ignored": lambda: graph_variant(rec, ignored_weight=0.0),
                  "graph_1hop": lambda: graph_variant(rec, hops=1),
                  "llm_only": lambda: llm_only_model(reranker),
                  "full": lambda: full_model(rec, reranker)}[name]()
            if reranker:
                reranker.stats = {k: 0 for k in reranker.stats}
            t0 = time.time()
            r = evaluate(sample, fn, k_values, recall_k, cold, rec.popularity, rec.category)
            r["seconds"] = round(time.time() - t0, 1)
            if name in ("full", "llm_only"):
                r["llm"] = {**reranker.stats, "fallback_rate": round(reranker.fallback_rate(), 4),
                            "client": dict(reranker.client.stats)}
            runs.append(r)
        mean = pd.DataFrame([r["all"] for r in runs]).mean().round(4).to_dict()
        results[name] = {"mean_over_seeds": mean, "runs": runs}
        print(f"{name:17s} " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))

    try:                                     # explanation fidelity (MLE-Core), if present
        from src.eval.fidelity import explanation_fidelity
    except ImportError:
        explanation_fidelity = None
    if "graph" in args.models and explanation_fidelity:
        sample = beh.sample(n=min(args.sample or 500, len(beh), 500), random_state=args.seeds[0])
        results["explanation_fidelity"] = explanation_fidelity(rec, sample)
        print("fidelity", results["explanation_fidelity"])

    out = ROOT / "results" / f"eval_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"split": args.split, "sample": args.sample, "seeds": args.seeds,
                   "config": {k: v for k, v in cfg.items() if isinstance(v, (dict, int, float))},
                   "results": results}, f, indent=2, default=str)
    print("Saved", out)
    if args.split == "test":
        record_test_run(args, out)


if __name__ == "__main__":
    main()
