"""Clean MIND small and build the train / validation / test split (DE + PL).

Output in data/processed/:
  news.pkl          one row per article: news_id, category, subcategory, title, abstract, entities
  entities.pkl      one row per entity: entity_id (Wikidata id), label, type
  behaviors_{train,val,test}.pkl  one row per impression: imp_id, user_id, time, history, candidates, labels
  stats.json        sizes, entity coverage, cold-start counts (Week 1 measures)

Split (no leakage): MIND train period minus its last day = train, its last day = validation,
MIND dev = test. Everything learned (graph edges, popularity, ignored lists) comes from train only.
"""
import json
from datetime import timedelta

import pandas as pd

from src.config import load_config

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
BEH_COLS = ["imp_id", "user_id", "time", "history", "impressions"]


def parse_entities(raw: str) -> list[dict]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [{"entity_id": e["WikidataId"], "label": e.get("Label", ""), "type": e.get("Type", "")}
            for e in items if e.get("WikidataId")]


def load_news(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    news = pd.concat([pd.read_csv(p, sep="\t", names=NEWS_COLS, quoting=3) for p in paths])
    news = news.drop_duplicates("news_id").dropna(subset=["news_id", "category"])
    news["title"] = news["title"].fillna("")
    news["abstract"] = news["abstract"].fillna("")
    ents, per_article = [], []
    for t, a in zip(news["title_entities"], news["abstract_entities"]):
        found = {}
        for e in parse_entities(t) + parse_entities(a):
            found[e["entity_id"]] = e
        per_article.append(sorted(found))
        ents.extend(found.values())
    news["entities"] = per_article
    news = news.drop(columns=["url", "title_entities", "abstract_entities"]).reset_index(drop=True)
    entities = pd.DataFrame(ents).drop_duplicates("entity_id").reset_index(drop=True)
    if entities.empty:
        entities = pd.DataFrame(columns=["entity_id", "label", "type"])
    return news, entities


def load_behaviors(path, known_news: set) -> pd.DataFrame:
    beh = pd.read_csv(path, sep="\t", names=BEH_COLS, quoting=3)
    n0 = len(beh)
    beh = beh.dropna(subset=["user_id", "time", "impressions"]).drop_duplicates("imp_id")
    beh["time"] = pd.to_datetime(beh["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    beh = beh.dropna(subset=["time"])
    beh["history"] = beh["history"].fillna("").map(
        lambda h: [n for n in h.split() if n in known_news])
    cands, labels = [], []
    for imp in beh["impressions"]:
        c, l = [], []
        for item in imp.split():
            nid, _, lab = item.rpartition("-")
            if nid in known_news and lab in ("0", "1"):
                c.append(nid)
                l.append(int(lab))
        cands.append(c)
        labels.append(l)
    beh["candidates"], beh["labels"] = cands, labels
    beh = beh[beh["candidates"].map(len) > 0].drop(columns=["impressions"])
    print(f"  {path.name}: {n0} rows read, {len(beh)} kept")
    return beh.reset_index(drop=True)


def main(config_path="configs/default.yaml") -> dict:
    cfg = load_config(config_path)
    raw, out = cfg["raw_dir"], cfg["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)

    news, entities = load_news([raw / "train" / "news.tsv", raw / "dev" / "news.tsv"])
    known = set(news["news_id"])
    train_full = load_behaviors(raw / "train" / "behaviors.tsv", known)
    test = load_behaviors(raw / "dev" / "behaviors.tsv", known)

    cut = train_full["time"].max().normalize() - timedelta(days=cfg["validation_days"] - 1)
    train = train_full[train_full["time"] < cut].reset_index(drop=True)
    val = train_full[train_full["time"] >= cut].reset_index(drop=True)

    news.to_pickle(out / "news.pkl")
    entities.to_pickle(out / "entities.pkl")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_pickle(out / f"behaviors_{name}.pkl")

    # Week 1 measures: size, entity coverage, cold start
    seen_train = set()
    for h, c in zip(train_full["history"], train_full["candidates"]):
        seen_train.update(h)
        seen_train.update(c)
    test_cands = set(n for c in test["candidates"] for n in c)
    cold = test_cands - seen_train
    stats = {
        "n_articles": int(len(news)),
        "n_entities": int(len(entities)),
        "share_articles_with_entity": round(float((news["entities"].map(len) > 0).mean()), 4),
        "n_users": int(pd.concat([train_full["user_id"], test["user_id"]]).nunique()),
        "n_impressions": {"train": len(train), "val": len(val), "test": len(test)},
        "validation_starts": str(cut),
        "test_candidate_articles": len(test_cands),
        "test_articles_never_in_train_period": len(cold),
        "share_test_impressions_with_cold_article": round(float(
            test["candidates"].map(lambda c: any(n in cold for n in c)).mean()), 4),
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
