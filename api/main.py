"""FastAPI backend (SE). Only the API touches the model and the database.

Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
PREDICT_MODE=mock (default in Week 2) or PREDICT_MODE=graph.
"""
import os
import uuid
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import load_config
from src.data.db import connect, log_interaction

app = FastAPI(title="News companion API")
CFG = load_config(os.getenv("RECSYS_CONFIG", "configs/default.yaml"))
DB = connect(CFG["db_path"])
DB.execute("CREATE TABLE IF NOT EXISTS UserTopics (user_id TEXT, topic TEXT, PRIMARY KEY (user_id, topic))")


@lru_cache
def predictor():
    from src.predict import Predictor
    return Predictor(mode=os.getenv("PREDICT_MODE", "mock"),
                     config_path=os.getenv("RECSYS_CONFIG", "configs/default.yaml"))


class Interaction(BaseModel):
    user_id: str
    article_id: str
    feedback: Literal["like", "dislike"]


class NewUser(BaseModel):
    topics: list[str] = Field(min_length=3)


def reactions(user_id):
    rows = DB.execute("SELECT article_id, feedback FROM Interactions WHERE user_id=?",
                      (user_id,)).fetchall()
    return [a for a, f in rows if f == "like"], [a for a, f in rows if f == "dislike"]


def article_info(ids):
    if not ids:
        return {}
    q = f"SELECT article_id, title, category FROM Articles WHERE article_id IN ({','.join('?' * len(ids))})"
    return {a: {"title": t, "category": c} for a, t, c in DB.execute(q, ids).fetchall()}


@app.get("/health")
def health():
    p = predictor()
    return {"status": "ok", "mode": p.mode, "graph_version": str(p.cfg["artefacts_dir"] /
            (p.cfg["artefacts_dir"] / "LATEST").read_text().strip()), "llm": None}


@app.get("/recommendations")
def recommendations(user_id: str, k: int = Query(10, ge=1, le=50), offset: int = Query(0, ge=0)):
    likes, dislikes = reactions(user_id)
    topics = [t for (t,) in DB.execute("SELECT topic FROM UserTopics WHERE user_id=?", (user_id,))]
    recs = predictor().predict(user_id, k, offset, likes, dislikes, topics or None)
    info = article_info([r["article_id"] for r in recs])
    return {"user_id": user_id, "items": [{**r, **info.get(r["article_id"], {})} for r in recs]}


@app.get("/user_profile")
def user_profile(user_id: str):
    p = predictor()
    likes, dislikes = reactions(user_id)
    history = list(p.known_history.get(user_id, [])) + likes
    prof = p.rec.build_profile(history, list(p.rec.ignored.get(user_id, [])) + dislikes)
    return {"user_id": user_id, "categories": prof.category_w, "diversity": round(prof.diversity, 3),
            "top_entities": prof.top_entities(p.rec.ent_labels),
            "summary": None}  # LLM text summary: Week 2 MLE-Ops / Week 3


@app.get("/user_history")
def user_history(user_id: str):
    rows = DB.execute("""SELECT i.article_id, a.title, i.feedback, i.timestamp FROM Interactions i
                         LEFT JOIN Articles a USING(article_id) WHERE user_id=? ORDER BY timestamp DESC""",
                      (user_id,)).fetchall()
    read = predictor().known_history.get(user_id, [])
    info = article_info(list(read))
    return {"reactions": [dict(zip(("article_id", "title", "feedback", "timestamp"), r)) for r in rows],
            "read": [{"article_id": a, **info.get(a, {})} for a in reversed(read)]}


@app.post("/log_interaction")
def post_interaction(x: Interaction):
    if not DB.execute("SELECT 1 FROM Articles WHERE article_id=?", (x.article_id,)).fetchone():
        raise HTTPException(404, "unknown article_id")
    return {"interaction_id": log_interaction(DB, x.user_id, x.article_id, x.feedback), "status": "stored"}


@app.get("/articles/search")
def search(q: str, k: int = 10):
    rows = DB.execute("SELECT article_id, title, category FROM Articles WHERE title LIKE ? LIMIT ?",
                      (f"%{q}%", k)).fetchall()
    return [dict(zip(("article_id", "title", "category"), r)) for r in rows]


@app.post("/users")
def new_user(body: NewUser):
    user_id = "new_" + uuid.uuid4().hex[:8]
    DB.execute("INSERT INTO Users(user_id) VALUES (?)", (user_id,))
    DB.executemany("INSERT INTO UserTopics VALUES (?, ?)", [(user_id, t) for t in body.topics])
    DB.commit()
    return {"user_id": user_id}
