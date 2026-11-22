"""Time and cost of one LLM call (MLE-Ops, closes the "LLM cost and speed" risk of Week 2).

    python scripts/llm_benchmark.py [config] --calls 5

For a few validation impressions it times one re-ranking call and one profile summary call without the
cache (a fresh temporary cache), then the same call again from the disk cache. It reports latency,
tokens, cost per call and the cost of the planned runs, in results/llm_benchmark.{json,md}.
Set the provider, model and prices in the llm section of the config, and the key in the environment.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, load_config  # noqa: E402
from src.graph.retrieval import GraphRecommender  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.llm.profile_summary import summarize  # noqa: E402
from src.llm.rerank import Reranker  # noqa: E402


def stats(calls):
    s = [c["seconds"] for c in calls]
    return {"n": len(calls), "median_s": round(float(np.median(s)), 3), "max_s": round(float(np.max(s)), 3),
            "mean_input_tokens": round(float(np.mean([c["input_tokens"] for c in calls])), 1),
            "mean_output_tokens": round(float(np.mean([c["output_tokens"] for c in calls])), 1),
            "mean_cost_usd": float(np.mean([c["cost_usd"] for c in calls]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--calls", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rec = GraphRecommender.from_config(cfg)
    beh = pd.read_pickle(cfg["processed_dir"] / "behaviors_val.pkl")
    beh = beh[beh["history"].map(len) > 0].sample(n=min(args.calls, len(beh)), random_state=1)

    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(**{**cfg["llm"], "cache_dir": tmp})
        rr = Reranker.from_config(cfg, rec, client)
        rows = {"rerank_no_cache": [], "rerank_cached": [], "summary_no_cache": [], "summary_cached": []}
        for row in beh.itertuples(index=False):
            profile = rec.build_profile(row.history, rec.ignored.get(row.user_id, []))
            cands = list(row.candidates)
            g = rec.score(profile, cands)
            for kind in ("no_cache", "cached"):
                rr.rerank(profile, cands, g)
                rows[f"rerank_{kind}"].append(dict(client.last_call))
                summarize(rec, profile, client)
                rows[f"summary_{kind}"].append(dict(client.last_call))
        res = {k: stats(v) for k, v in rows.items()}

    per_imp = res["rerank_no_cache"]["mean_cost_usd"]
    res["setup"] = {"provider": client.provider, "model": client.model,
                    "price_input_per_mtok": client.price_in, "price_output_per_mtok": client.price_out,
                    "top_n": cfg["rerank"]["top_n"], "fallback_rate": round(rr.fallback_rate(), 4)}
    res["planned_runs_usd"] = {
        "1000_impressions": round(1000 * per_imp, 4),
        "eval_3_seeds_x_2000": round(6000 * per_imp, 4),
        "fusion_training_4000": round(4000 * per_imp, 4)}
    res["eval_3_seeds_x_2000_minutes_no_cache"] = round(6000 * res["rerank_no_cache"]["median_s"] / 60, 1)

    out = ROOT / cfg.get("results_dir", "results") / "llm_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    s = res["setup"]
    md = [f"# LLM call time and cost ({s['provider']} {s['model']}, top {s['top_n']})", "",
          "| Call | Median time (s) | Max (s) | Input tokens | Output tokens | Cost per call ($) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for k in ("rerank_no_cache", "rerank_cached", "summary_no_cache", "summary_cached"):
        r = res[k]
        md.append(f"| {k.replace('_', ' ')} | {r['median_s']} | {r['max_s']} | {r['mean_input_tokens']} | "
                  f"{r['mean_output_tokens']} | {r['mean_cost_usd']:.5f} |")
    p = res["planned_runs_usd"]
    md += ["", f"Re-ranking 1,000 impressions costs about ${p['1000_impressions']}, the evaluation "
           f"(3 seeds x 2,000) about ${p['eval_3_seeds_x_2000']} and "
           f"{res['eval_3_seeds_x_2000_minutes_no_cache']} minutes without cache. A cached call costs "
           "nothing, so reruns and the website use the cache."]
    (out.parent / "llm_benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
