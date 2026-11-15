"""SQLite database (DE): schema, loading of articles and entities, interactions."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS Users (user_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS Articles (
    article_id TEXT PRIMARY KEY, category TEXT, subcategory TEXT, title TEXT, abstract TEXT);
CREATE TABLE IF NOT EXISTS Entities (entity_id TEXT PRIMARY KEY, label TEXT, type TEXT);
CREATE TABLE IF NOT EXISTS ArticleEntities (
    article_id TEXT REFERENCES Articles(article_id),
    entity_id TEXT REFERENCES Entities(entity_id),
    PRIMARY KEY (article_id, entity_id));
CREATE TABLE IF NOT EXISTS Interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES Users(user_id),
    article_id TEXT NOT NULL REFERENCES Articles(article_id),
    feedback TEXT NOT NULL CHECK (feedback IN ('like', 'dislike')),
    timestamp TEXT NOT NULL,
    UNIQUE (user_id, article_id));
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def log_interaction(con, user_id: str, article_id: str, feedback: str) -> int:
    """Store a reaction; a second click on the same article replaces the first."""
    con.execute("INSERT OR IGNORE INTO Users(user_id) VALUES (?)", (user_id,))
    ts = datetime.now(timezone.utc).isoformat()
    cur = con.execute(
        """INSERT INTO Interactions(user_id, article_id, feedback, timestamp) VALUES (?, ?, ?, ?)
           ON CONFLICT(user_id, article_id) DO UPDATE SET feedback=excluded.feedback,
           timestamp=excluded.timestamp RETURNING interaction_id""",
        (user_id, article_id, feedback, ts))
    interaction_id = cur.fetchone()[0]
    con.commit()
    return interaction_id


def main(config_path="configs/default.yaml") -> None:
    cfg = load_config(config_path)
    proc = cfg["processed_dir"]
    news = pd.read_pickle(proc / "news.pkl")
    entities = pd.read_pickle(proc / "entities.pkl")
    users = pd.concat([pd.read_pickle(proc / f"behaviors_{s}.pkl")["user_id"]
                       for s in ("train", "val", "test")]).unique()
    con = connect(cfg["db_path"])
    con.executemany("INSERT OR IGNORE INTO Users VALUES (?)", [(u,) for u in users])
    con.executemany("INSERT OR IGNORE INTO Articles VALUES (?,?,?,?,?)",
                    news[["news_id", "category", "subcategory", "title", "abstract"]]
                    .itertuples(index=False))
    con.executemany("INSERT OR IGNORE INTO Entities VALUES (?,?,?)",
                    entities[["entity_id", "label", "type"]].itertuples(index=False))
    pairs = [(n, e) for n, ents in zip(news["news_id"], news["entities"]) for e in ents]
    con.executemany("INSERT OR IGNORE INTO ArticleEntities VALUES (?,?)", pairs)
    con.commit()
    for t in ("Users", "Articles", "Entities", "ArticleEntities", "Interactions"):
        print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    con.close()


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
