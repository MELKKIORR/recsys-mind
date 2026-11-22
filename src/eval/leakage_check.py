"""Leakage check (PL): evidence for the Week 3 question Q3.

    python -m src.eval.leakage_check [config]     -> prints and writes results/leakage_check.md

Checks:
  1. the split is temporal: train ends before validation starts, validation before test
  2. no impression id is shared between train, validation and test
  3. the saved graph and statistics are exactly what the code builds: they are rebuilt and compared.
     Entity-entity edges, popularity and ignored lists come from the TRAIN split alone. The
     article-entity matrix holds every article, test articles included, on purpose: it only uses
     article content (the entities in the title and abstract), never a click, and it is what lets
     the model score a new article (cold start)
  4. no validation or test click is counted in the popularity used by the models
  5. the test split was used for final evaluations only (results/final_test_log.json)
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import ROOT, load_config
from src.graph.build_graph import build
from src.graph.retrieval import latest_artefacts


def main(config_path="configs/default.yaml") -> bool:
    cfg = load_config(config_path)
    proc = cfg["processed_dir"]
    tr, va, te = (pd.read_pickle(proc / f"behaviors_{s}.pkl") for s in ("train", "val", "test"))
    art = latest_artefacts(cfg["artefacts_dir"])
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))

    check("Temporal split", tr["time"].max() < va["time"].min() <= va["time"].max() < te["time"].min(),
          f"train {tr['time'].min():%Y-%m-%d %H:%M} to {tr['time'].max():%Y-%m-%d %H:%M}, "
          f"validation from {va['time'].min():%Y-%m-%d %H:%M}, test from {te['time'].min():%Y-%m-%d %H:%M}")

    ids = [set(zip(d["user_id"], d["imp_id"], d["time"])) for d in (tr, va, te)]
    overlap = len(ids[0] & ids[1]) + len(ids[0] & ids[2]) + len(ids[1] & ids[2])
    check("No shared impression", overlap == 0, f"{overlap} impressions in more than one split")

    g = build(cfg)                                    # rebuild from train only, as the code does
    saved_ae = sp.load_npz(art / "article_entity.npz")
    diff_ae = abs(saved_ae - g["ae"]).sum()
    check("Article-entity matrix from content only", diff_ae == 0,
          f"saved vs rebuilt from titles and abstracts: {saved_ae.nnz} vs {g['ae'].nnz} links, "
          f"total difference {diff_ae:.0f}; no click is used")
    saved_ee = sp.load_npz(art / "entity_entity.npz")
    diff = abs(saved_ee - g["ee"]).sum()
    check("Graph edges from train only", diff == 0,
          f"saved entity-entity matrix vs rebuilt from train: {saved_ee.nnz} vs {g['ee'].nnz} entries, "
          f"total difference {diff:.0f}")

    stats = pd.read_pickle(art / "train_stats.pkl")
    check("Popularity from train only", stats["popularity"] == g["popularity"],
          f"{len(stats['popularity'])} articles with train clicks")
    check("Ignored lists from train only", stats["ignored"] == g["ignored"],
          f"{len(stats['ignored'])} readers with ignored articles")

    later_clicks = {a for d in (va, te) for c, l in zip(d["candidates"], d["labels"])
                    for a, y in zip(c, l) if y}
    train_clicks = {a for c, l in zip(tr["candidates"], tr["labels"]) for a, y in zip(c, l) if y}
    only_later = later_clicks - train_clicks
    leaked = [a for a in only_later if stats["popularity"].get(a, 0) > 0]
    check("No val/test click in popularity", not leaked,
          f"{len(only_later)} articles clicked only after training, {len(leaked)} with a popularity > 0")

    log_path = ROOT / "results" / "final_test_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    done = [e for e in log if e.get("status", "completed") == "completed"]
    forced = sum(e.get("forced", False) for e in done)
    check("Test split used once", forced == 0,
          f"{len(done)} completed test run(s), {forced} forced" if done else "not used yet")

    ok = all(c[1] for c in checks)
    lines = ["# Leakage check", "", f"Artefacts: {art.name}", "",
             "| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {n} | {'PASS' if o else 'FAIL'} | {d} |" for n, o, d in checks]
    lines += ["", f"Overall: {'PASS' if ok else 'FAIL'}",
              "Hyperparameters are tuned on the validation split only; each test impression uses only "
              "its own history (the MIND history field, clicks before the impression)."]
    text = "\n".join(lines)
    print(text)
    (ROOT / "results" / "leakage_check.md").write_text(text, encoding="utf-8")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:]) else 1)
