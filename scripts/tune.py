"""Hyper-parameter search on validation, then 3 draws of 2,000 impressions (MLE-Ops, Week 3).

    python scripts/tune.py [config] --sample 1000 --final-sample 2000 --seeds 1 2 3

Everything runs on the validation split, on the "holdout" readers only (half of the validation readers,
fixed by a hash of the user id). The other half, the "fit" readers, trains the fusion layer. The test
split is never read: the single final test run is done by the PL with src.eval.evaluate --final.

1. Graph: one parameter at a time (hops, damping, recency decay, ignored weight, category weight), the
   others at their best value so far, chosen on NDCG@10 of `--sample` impressions (seed 1).
2. Fusion: fixed rule weight 0 to 1, then the learned layer for 3 values of L2 (trained on fit readers).
3. Final: random, popular, graph, full (fixed rule) and full (learned layer) with the best values,
   on 3 draws of `--final-sample` impressions, mean and standard deviation.

Output: results/tuning_val.{json,md}, results/figures/tuning_val.png, results/figures/fusion_loss.png,
results/fusion_weights.json, results/final_val_3seeds.json, configs/tuned.yaml.
"""
import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config  # noqa: E402
from src.eval.evaluate import evaluate, popular_model, random_model  # noqa: E402
from src.graph.retrieval import GraphRecommender  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.llm.fusion import FusionLayer, plot_curves, train_from_config, val_half  # noqa: E402
from src.llm.rerank import Reranker  # noqa: E402

GRID = {
    "hops": [1, 2, 3],
    "damping": [0.5, 0.65, 0.75, 0.85],
    "recency_decay": [0.7, 0.8, 0.9, 1.0],
    "ignored_weight": [0.0, 0.25, 0.5, 1.0],
    "category_weight": [0.0, 0.15, 0.3, 0.5],
}
FUSION_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
L2_VALUES = [0.0, 1e-3, 1e-2]
METRIC = "ndcg@10"


def variant(rec, params):
    r = copy.copy(rec)
    for k, v in params.items():
        setattr(r, k, v)
    return r


def graph_fn(rec):
    return lambda row: rec.forward(row.history, row.candidates, ignored=rec.ignored.get(row.user_id, []),
                                   with_paths=False)[0]


def full_fn(rec, reranker):
    """Graph + LLM re-ranking. Evidence paths are only computed for the top N sent to the LLM."""
    def score(row):
        profile = rec.build_profile(row.history, rec.ignored.get(row.user_id, []))
        cands = list(row.candidates)
        g = rec.score(profile, cands)
        top = set(np.argsort(-g, kind="stable")[:reranker.top_n])
        paths = [rec.evidence_path(profile, c) if i in top else None for i, c in enumerate(cands)]
        return reranker.rerank(profile, cands, g, paths)["scores"]
    return score


def run(beh, fn, cfg, cold):
    return evaluate(beh, fn, cfg["eval"]["k_values"], cfg["eval"]["recall_k"], cold)


def plot_search(search, fusion_rule, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(search) + ["fusion_weight"]
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 2.8), sharey=True)
    for ax, name in zip(axes, names):
        pts = search[name] if name in search else fusion_rule
        ax.plot([p["value"] for p in pts], [p[METRIC] for p in pts], marker="o")
        best = max(pts, key=lambda p: p[METRIC])
        ax.scatter([best["value"]], [best[METRIC]], color="tab:red", zorder=3)
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("validation NDCG@10")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--sample", type=int, default=1000, help="impressions per search point")
    ap.add_argument("--final-sample", type=int, default=2000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--fusion-train", type=int, default=3000)
    ap.add_argument("--skip-llm", action="store_true", help="only the graph search and graph baselines")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("results_dir", "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = cfg["processed_dir"]
    val = pd.read_pickle(proc / "behaviors_val.pkl")
    hold = val_half(val, "holdout")
    train = pd.read_pickle(proc / "behaviors_train.pkl")
    seen = {n for h, c in zip(train["history"], train["candidates"]) for n in (*h, *c)}
    cold = {n for c in hold["candidates"] for n in c} - seen
    search_beh = hold.sample(n=min(args.sample, len(hold)), random_state=args.seeds[0])
    print(f"Validation: {len(val)} impressions, holdout readers {len(hold)}, search sample {len(search_beh)}")

    rec = GraphRecommender.from_config(cfg)
    best = {k: cfg["retrieval"][k] for k in GRID}
    search, t0 = {}, time.time()

    # 1. graph ------------------------------------------------------------------------------------
    for name, values in GRID.items():
        search[name] = []
        for v in values:
            r = run(search_beh, graph_fn(variant(rec, {**best, name: v})), cfg, cold)
            search[name].append({"value": v, **r["all"]})
            print(f"{name:16s} {v:<5}  NDCG@10 {r['all'][METRIC]:.4f}  AUC {r['all']['auc']:.4f}")
        best[name] = max(search[name], key=lambda p: p[METRIC])["value"]
        print(f"  -> {name} = {best[name]}  ({time.time() - t0:.0f} s)")
    rec_best = variant(rec, best)

    # 2. fusion -----------------------------------------------------------------------------------
    fusion_rule, fusion_learned, best_w, best_l2 = [], [], cfg["rerank"]["fusion_weight"], None
    client = LLMClient.from_config(cfg)
    rr_args = (cfg["rerank"]["top_n"], 0.5, cfg["rerank"]["min_valid_share"], cfg["llm"]["prompt_version"])
    layer, layer_hist = None, None
    if not args.skip_llm:
        for w in FUSION_WEIGHTS:
            rr = Reranker(rec_best, client, rr_args[0], w, *rr_args[2:])
            r = run(search_beh, full_fn(rec_best, rr), cfg, cold)
            fusion_rule.append({"value": w, **r["all"], "fallback_rate": round(rr.fallback_rate(), 4)})
            print(f"fusion_weight    {w:<5}  NDCG@10 {r['all'][METRIC]:.4f}  fallback {rr.fallback_rate():.3f}")
        best_w = max(fusion_rule, key=lambda p: p[METRIC])["value"]
        cfg_best = copy.deepcopy(cfg)
        cfg_best["retrieval"].update(best)
        best_score = -1
        for l2 in L2_VALUES:
            lay, hist, info = train_from_config(cfg_best, args.fusion_train, args.sample, args.seeds[0],
                                                rec=rec_best, client=client, log=lambda *_: None, l2=l2)
            rr = Reranker(rec_best, client, rr_args[0], best_w, *rr_args[2:], fusion=lay)
            r = run(search_beh, full_fn(rec_best, rr), cfg, cold)
            fusion_learned.append({"value": l2, **r["all"], "epochs": len(hist) - 1,
                                   "best_val_loss": round(min(h["val_loss"] for h in hist), 5), **info})
            print(f"learned l2       {l2:<5}  NDCG@10 {r['all'][METRIC]:.4f}  epochs {len(hist) - 1}")
            if r["all"][METRIC] > best_score:
                best_score, best_l2, layer, layer_hist, layer_info = r["all"][METRIC], l2, lay, hist, info
        wpath = ROOT / cfg["rerank"].get("fusion_weights", "results/fusion_weights.json")
        layer.save(wpath, training={**layer_info, "l2": best_l2, "lr": 0.01, "epochs_run": len(layer_hist) - 1,
                                    "graph_params": best}, history=layer_hist)
        plot_curves(layer_hist, out_dir / "figures" / "fusion_loss.png",
                    f"Fusion layer (L2 = {best_l2}): loss on fit and holdout readers")

    plot_search(search, fusion_rule or [{"value": best_w, METRIC: float("nan")}],
                out_dir / "figures" / "tuning_val.png")
    tuning = {"date": f"{datetime.now():%Y-%m-%d %H:%M}", "sample": len(search_beh), "seed": args.seeds[0],
              "metric": METRIC, "graph_search": search, "best_graph": best,
              "fusion_rule": fusion_rule, "best_fusion_weight": best_w,
              "fusion_learned_l2": fusion_learned, "best_l2": best_l2,
              "llm": {"provider": client.provider, "model": client.model, **client.stats}}
    (out_dir / "tuning_val.json").write_text(json.dumps(tuning, indent=2, default=float))

    tuned = yaml.safe_load((ROOT / args.config).read_text()) if not Path(args.config).is_absolute() \
        else yaml.safe_load(Path(args.config).read_text())
    tuned["retrieval"].update(best)
    tuned["rerank"]["fusion_weight"] = best_w
    if layer is not None:
        tuned["rerank"]["fusion"] = "learned"
    tuned_path = ROOT / "configs" / ("tuned.yaml" if "test" not in args.config else "tuned_test.yaml")
    tuned_path.write_text("# Written by scripts/tune.py on the validation split (holdout readers)\n"
                          + yaml.safe_dump(tuned, sort_keys=False))

    # 3. final: 3 draws -------------------------------------------------------------------------------
    models = {"random": None, "popular": None, "graph (default)": None, "graph (tuned)": None}
    if not args.skip_llm:
        models["full, fixed rule"] = None
        models["full, learned fusion"] = None
    final = {}
    for name in models:
        runs = []
        for seed in args.seeds:
            beh = hold.sample(n=min(args.final_sample, len(hold)), random_state=seed)
            client.reset_stats()
            rr = None
            fn = {"random": lambda: random_model(seed),
                  "popular": lambda: popular_model(rec.popularity),
                  "graph (default)": lambda: graph_fn(rec),
                  "graph (tuned)": lambda: graph_fn(rec_best)}.get(name)
            if fn is None:
                rr = Reranker(rec_best, client, rr_args[0], best_w, *rr_args[2:],
                              fusion=layer if "learned" in name else None)
                fn = lambda: full_fn(rec_best, rr)  # noqa: E731
            t = time.time()
            r = run(beh, fn(), cfg, cold)
            r["seconds"] = round(time.time() - t, 1)
            if rr is not None:
                r["llm"] = {"fallback_rate": round(rr.fallback_rate(), 4), **rr.stats, "client": dict(client.stats)}
            runs.append(r)
        df = pd.DataFrame([r["all"] for r in runs])
        final[name] = {"mean": df.mean().round(4).to_dict(), "std": df.std(ddof=0).round(4).to_dict(),
                       "runs": runs}
        m = final[name]["mean"]
        print(f"{name:22s} NDCG@10 {m['ndcg@10']:.4f}  Recall@20 {m.get('recall@20', float('nan')):.4f}  "
              f"AUC {m['auc']:.4f}")
    (out_dir / "final_val_3seeds.json").write_text(json.dumps(
        {"split": "val (holdout readers)", "sample": args.final_sample, "seeds": args.seeds,
         "graph_params": best, "fusion_weight": best_w, "fusion_l2": best_l2, "results": final},
        indent=2, default=float))

    # markdown summary ------------------------------------------------------------------------------
    rk = f"recall@{cfg['eval']['recall_k']}"
    md = ["# Week 3 tuning on validation (MLE-Ops)", "",
          f"Validation split, holdout readers only, {len(search_beh)} impressions per point (seed "
          f"{args.seeds[0]}), metric NDCG@10. LLM: {client.provider} {client.model}.", "",
          "| Parameter | Values tried (NDCG@10) | Best |", "| --- | --- | --- |"]
    for name, pts in search.items():
        md.append(f"| {name} | " + ", ".join(f"{p['value']}: {p[METRIC]:.4f}" for p in pts) + f" | {best[name]} |")
    if fusion_rule:
        md.append("| fusion_weight (rule) | " + ", ".join(f"{p['value']}: {p[METRIC]:.4f}" for p in fusion_rule)
                  + f" | {best_w} |")
        md.append("| L2 of the learned layer | " + ", ".join(f"{p['value']}: {p[METRIC]:.4f}"
                                                         for p in fusion_learned) + f" | {best_l2} |")
    md += ["", f"## Final: {len(args.seeds)} draws of {args.final_sample} impressions (seeds {args.seeds})", "",
           f"| Model | NDCG@10 | {rk} | AUC | MRR | NDCG@5 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, r in final.items():
        cells = [f"{r['mean'][k]:.4f} ± {r['std'][k]:.4f}" for k in ("ndcg@10", rk, "auc", "mrr", "ndcg@5")]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md += ["", "Figures: figures/tuning_val.png (one parameter at a time), figures/fusion_loss.png "
           "(train and validation loss of the fusion layer). Best values written to "
           f"configs/{tuned_path.name}."]
    (out_dir / "tuning_val.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
