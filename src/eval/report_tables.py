"""Tables and figures for the Week 3 report (PL), from the files written by src.eval.evaluate.

    python -m src.eval.report_tables                  # latest val and latest test results
    python -m src.eval.report_tables results/eval_test_XXXX.json

Writes results/report_tables.md and results/figures/*.png. Values are the mean over the seeds,
with the standard deviation when there is more than one seed.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import ROOT  # noqa: E402

ORDER = ["random", "popular", "graph", "graph_no_ignored", "graph_1hop", "llm_only", "full"]
LABEL = {"random": "Random", "popular": "Most popular", "graph": "Graph only",
         "graph_no_ignored": "Graph, no ignored signal", "graph_1hop": "Graph, 1 hop",
         "llm_only": "LLM only (prompt)", "full": "Full system (graph + LLM)"}


def stat(runs, key, part="all"):
    vals = [r.get(part, {}).get(key) for r in runs]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else None)


def fmt(m, s, digits=3):
    if m is None:
        return "-"
    return f"{m:.{digits}f}" + (f" ± {s:.{digits}f}" if s is not None else "")


def latest(split):
    files = sorted((ROOT / "results").glob(f"eval_{split}_*.json"))
    return files[-1] if files else None


def tables(path: Path) -> list:
    data = json.loads(path.read_text())
    res, split = data["results"], data["split"]
    models = [m for m in ORDER if m in res]
    lines = [f"## {split.capitalize()} split ({path.name}, sample {data['sample'] or 'all'}, "
             f"seeds {data['seeds']})", "",
             "| Model | NDCG@10 | Recall@20 | AUC | MRR | Cold-start NDCG@10 | NDCG@10, < 5 clicks | "
             "Top 10 popularity | Top 10 categories |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in models:
        runs = res[m]["runs"]
        lines.append("| " + " | ".join([
            LABEL[m], fmt(*stat(runs, "ndcg@10")), fmt(*stat(runs, "recall@20")),
            fmt(*stat(runs, "auc")), fmt(*stat(runs, "mrr")), fmt(*stat(runs, "cold_ndcg@10")),
            fmt(*stat(runs, "ndcg@10", "short_history")),
            fmt(*stat(runs, "top10_mean_popularity"), digits=1),
            fmt(*stat(runs, "top10_n_categories"), digits=2)]) + " |")
    extra = []
    for m in ("full", "llm_only"):
        if m in res and res[m]["runs"][0].get("llm"):
            llm = res[m]["runs"][0]["llm"]
            extra.append(f"{LABEL[m]}: fallback rate {llm['fallback_rate']:.1%}, "
                         f"{llm['unknown_ids']} unknown ids, {llm['partial']} partial answers, "
                         f"{llm['client']['calls']} LLM calls, {llm['client']['cache_hits']} from cache.")
    if "explanation_fidelity" in res:
        f = res["explanation_fidelity"]
        extra.append(f"Explanation fidelity: {f['share_explained_that_drop']:.1%} of the "
                     f"{f['explained_in_top_k']} explained top-10 articles drop when the entity of their "
                     f"evidence path is removed (mean loss {f['mean_rank_loss_when_drop']} ranks).")
    if "graph" in res and "graph_no_ignored" in res:
        a, _ = stat(res["graph"]["runs"], "ndcg@10")
        b, _ = stat(res["graph_no_ignored"]["runs"], "ndcg@10")
        extra.append(f"Ignored signal ablation: NDCG@10 {a:.3f} with it, {b:.3f} without it "
                     f"({(a - b) / b:+.1%}).")
    lines += [""] + extra + [""]
    figure(res, models, split)
    return lines


def figure(res, models, split):
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    keys = [("ndcg@10", "NDCG@10", "all"), ("recall@20", "Recall@20", "all"),
            ("cold_ndcg@10", "Cold-start NDCG@10", "all")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, title, part) in zip(axes, keys):
        means = [stat(res[m]["runs"], key, part) for m in models]
        y = [m[0] or 0 for m in means]
        err = [m[1] or 0 for m in means]
        colors = ["#b0b0b0" if m in ("random", "popular") else "#1f6f8b" if m == "full" else "#7fb3c8"
                  for m in models]
        ax.set_title(title, fontsize=10)
        if all(m[0] is None for m in means):
            ax.text(0.5, 0.5, "no cold-start article\nin this split", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            continue
        ax.bar(range(len(models)), y, yerr=err, color=colors, capsize=3)
        ax.set_xticks(range(len(models)), [LABEL[m] for m in models], fontsize=7, rotation=35,
                      ha="right")
        for i, v in enumerate(y):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"Baselines and ablations, {split} split", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"ablations_{split}.png", dpi=160)
    plt.close(fig)


def main(*paths):
    files = [Path(p) for p in paths] or [p for p in (latest("val"), latest("test")) if p]
    if not files:
        raise SystemExit("No results yet: run python -m src.eval.evaluate first.")
    lines = ["# Results tables", ""]
    for p in files:
        lines += tables(p)
    text = "\n".join(lines)
    print(text)
    (ROOT / "results" / "report_tables.md").write_text(text, encoding="utf-8")
    print("Figures in", ROOT / "results" / "figures")


if __name__ == "__main__":
    main(*sys.argv[1:])
