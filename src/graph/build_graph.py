"""Build the knowledge graph and the training-period statistics (PL), then save the artefacts.

Nodes: articles, entities, categories. Stored as SciPy sparse matrices:
  article_entity.npz   articles x entities (1 if the article mentions the entity), from content only
  entity_entity.npz    entities x entities, weight = number of TRAIN articles where both appear
Plus: popularity (clicks in train impressions), ignored articles per user (shown in train, never clicked).
No validation or test interaction is used here.
"""
import json
import subprocess
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import ROOT, load_config


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def build(cfg: dict) -> dict:
    proc = cfg["processed_dir"]
    news = pd.read_pickle(proc / "news.pkl")
    entities = pd.read_pickle(proc / "entities.pkl")
    train = pd.read_pickle(proc / "behaviors_train.pkl")

    art_idx = {n: i for i, n in enumerate(news["news_id"])}
    ent_idx = {e: i for i, e in enumerate(entities["entity_id"])}
    rows, cols = [], []
    for n, ents in zip(news["news_id"], news["entities"]):
        for e in ents:
            rows.append(art_idx[n])
            cols.append(ent_idx[e])
    ae = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                       shape=(len(art_idx), len(ent_idx)))

    # Entity co-occurrence from articles seen in the training period only
    train_articles = set()
    for h, c in zip(train["history"], train["candidates"]):
        train_articles.update(h)
        train_articles.update(c)
    mask = np.zeros(len(art_idx), dtype=np.float32)
    mask[[art_idx[a] for a in train_articles]] = 1.0
    ae_train = sp.diags(mask) @ ae
    ee = (ae_train.T @ ae_train).tocsr()
    ee.setdiag(0)
    ee.data[ee.data < cfg["graph"]["min_cooccurrence"]] = 0
    ee.eliminate_zeros()

    # Popularity and ignored articles from train impressions
    clicks = Counter()
    ignored = defaultdict(list)
    max_imp = cfg["retrieval"]["max_ignored_impressions"]
    for user, cands, labels in zip(train["user_id"], train["candidates"], train["labels"]):
        for n, l in zip(cands, labels):
            if l:
                clicks[n] += 1
        ignored[user].append([n for n, l in zip(cands, labels) if not l])
    ignored = {u: sorted({n for imp in imps[-max_imp:] for n in imp}) for u, imps in ignored.items()}
    clicked_by_user = defaultdict(set)
    for user, cands, labels in zip(train["user_id"], train["candidates"], train["labels"]):
        clicked_by_user[user].update(n for n, l in zip(cands, labels) if l)
    ignored = {u: [n for n in ns if n not in clicked_by_user[u]] for u, ns in ignored.items()}

    return {"news": news, "entities": entities, "art_idx": art_idx, "ent_idx": ent_idx,
            "ae": ae, "ee": ee, "popularity": dict(clicks), "ignored": ignored}


def save(g: dict, cfg: dict) -> str:
    out = cfg["artefacts_dir"] / f"{date.today().isoformat()}_{git_commit()}"
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "article_entity.npz", g["ae"])
    sp.save_npz(out / "entity_entity.npz", g["ee"])
    g["news"].to_pickle(out / "articles.pkl")
    g["entities"].to_pickle(out / "entities.pkl")
    pd.to_pickle({"popularity": g["popularity"], "ignored": g["ignored"]}, out / "train_stats.pkl")
    with open(out / "config.json", "w") as f:
        json.dump({k: (str(v) if not isinstance(v, (dict, list, int, float)) else v)
                   for k, v in cfg.items()}, f, indent=2, default=str)
    (cfg["artefacts_dir"] / "LATEST").write_text(out.name)
    return str(out)


def main(config_path="configs/default.yaml") -> None:
    cfg = load_config(config_path)
    g = build(cfg)
    path = save(g, cfg)
    print(f"Graph: {g['ae'].shape[0]} articles, {g['ae'].shape[1]} entities, "
          f"{g['ae'].nnz} article-entity edges, {g['ee'].nnz // 2} entity-entity edges")
    print("Artefacts saved in", path)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
