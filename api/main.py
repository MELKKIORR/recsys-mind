"""FastAPI backend (SE). Only the API touches the model and the database.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
PREDICT_MODE=graph (default: graph retrieval + LLM re-ranking) or PREDICT_MODE=mock (popular articles).
LLM_CACHE_ONLY=1 (default): the LLM re-ranking is read from the disk cache, never called while a reader
waits; on a cache miss the graph ranking is served and the cache is filled in the background.
Every response carries its server time in the X-Response-Time-ms header; GET /latency summarises them.
"""
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.config import load_config
from src.data.db import connect, log_interaction

CONFIG = os.getenv("RECSYS_CONFIG", "configs/default.yaml")
CFG = load_config(CONFIG)
DB = connect(CFG["db_path"])
DB.execute("CREATE TABLE IF NOT EXISTS UserTopics (user_id TEXT, topic TEXT, PRIMARY KEY (user_id, topic))")
DB_LOCK = threading.Lock()           # one SQLite connection shared by the worker threads


@lru_cache
def predictor():
    from src.predict import Predictor
    return Predictor(mode=os.getenv("PREDICT_MODE", "graph"), config_path=CONFIG,
                     cache_only=os.getenv("LLM_CACHE_ONLY", "1") != "0")


@asynccontextmanager
async def lifespan(_):
    predictor()                      # load the graph at start-up, not during the first request
    yield


app = FastAPI(title="News companion API", lifespan=lifespan)

# Response times ------------------------------------------------------------------------------------
LATENCY = defaultdict(lambda: deque(maxlen=1000))


@app.middleware("http")
async def response_time(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    LATENCY[request.url.path].append(ms)
    return response


@app.get("/latency")
def latency():
    """Server time per endpoint over the last 1000 calls, in milliseconds."""
    out = {}
    for path, values in LATENCY.items():
        v = np.array(values)
        out[path] = {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1),
                     "p95": round(float(np.percentile(v, 95)), 1), "max": round(float(v.max()), 1),
                     "share_under_2s": round(float((v < 2000).mean()), 3)}
    return out


# Helpers -------------------------------------------------------------------------------------------
def query(sql, params=()):
    with DB_LOCK:
        return DB.execute(sql, params).fetchall()


class Interaction(BaseModel):
    user_id: str
    article_id: str
    feedback: Literal["like", "dislike"]


class NewUser(BaseModel):
    topics: list[str] = Field(min_length=3)


def reactions(user_id):
    rows = query("SELECT article_id, feedback FROM Interactions WHERE user_id=? ORDER BY timestamp",
                 (user_id,))
    return [a for a, f in rows if f == "like"], [a for a, f in rows if f == "dislike"]


def topics_of(user_id):
    return [t for (t,) in query("SELECT topic FROM UserTopics WHERE user_id=?", (user_id,))]


def article_info(ids):
    if not ids:
        return {}
    q = f"SELECT article_id, title, category, abstract FROM Articles WHERE article_id IN ({','.join('?' * len(ids))})"
    return {a: {"title": t, "category": c, "abstract": ab} for a, t, c, ab in query(q, ids)}


WARMING, WARM_LOCK = set(), threading.Lock()


def warm_llm_cache(user_id, k, offset, likes, dislikes, topics):
    """Background task: call the LLM for this reader so the next request is served from the cache."""
    try:
        predictor().predict(user_id, k, offset, likes, dislikes, topics, live_llm=True)
    finally:
        with WARM_LOCK:
            WARMING.discard(user_id)


# Endpoints -----------------------------------------------------------------------------------------
@app.get("/health")
def health():
    p = predictor()
    llm = p.cfg.get("llm", {})
    return {"status": "ok", "mode": p.mode,
            "graph_version": (p.cfg["artefacts_dir"] / "LATEST").read_text().strip(),
            "llm": {"provider": llm.get("provider"), "model": llm.get("model"),
                    "cache_only": p.reranker.client.cache_only,
                    "serving": p.reranker.client.stats, "background": p.live_reranker.client.stats,
                    "rerank": p.reranker.stats}}


@app.get("/recommendations")
def recommendations(user_id: str, background: BackgroundTasks, k: int = Query(10, ge=1, le=50),
                    offset: int = Query(0, ge=0)):
    likes, dislikes = reactions(user_id)
    topics = topics_of(user_id) or None
    recs = predictor().predict(user_id, k, offset, likes, dislikes, topics)
    status = recs[0]["llm_status"] if recs else "not_used"
    if status == "fallback_cache_miss":
        with WARM_LOCK:
            start = user_id not in WARMING
            WARMING.add(user_id)
        if start:
            background.add_task(warm_llm_cache, user_id, k, offset, likes, dislikes, topics)
    info = article_info([r["article_id"] for r in recs])
    return {"user_id": user_id, "llm_status": status, "mode": predictor().mode,
            "items": [{**r, **info.get(r["article_id"], {})} for r in recs]}


@app.get("/user_profile")
def user_profile(user_id: str):
    p = predictor()
    likes, dislikes = reactions(user_id)
    history = list(p.known_history.get(user_id, [])) + likes
    prof = p.rec.build_profile(history, list(p.rec.ignored.get(user_id, [])) + dislikes,
                               topics_of(user_id) or None)
    return {"user_id": user_id, "n_read": len(history), "n_likes": len(likes),
            "n_dislikes": len(dislikes), "topics": topics_of(user_id),
            "categories": prof.category_w, "diversity": round(prof.diversity, 3),
            "top_entities": prof.top_entities(p.rec.ent_labels),
            "summary": None}  # LLM text summary: MLE-Ops, Week 3


@app.get("/user_history")
def user_history(user_id: str):
    rows = query("""SELECT i.interaction_id, i.article_id, a.title, i.feedback, i.timestamp
                    FROM Interactions i LEFT JOIN Articles a USING(article_id)
                    WHERE user_id=? ORDER BY timestamp DESC""", (user_id,))
    read = predictor().known_history.get(user_id, [])
    info = article_info(list(read))
    return {"reactions": [dict(zip(("interaction_id", "article_id", "title", "feedback", "timestamp"), r))
                          for r in rows],
            "read": [{"article_id": a, **info.get(a, {})} for a in reversed(read)]}


@app.post("/log_interaction")
def post_interaction(x: Interaction):
    if not query("SELECT 1 FROM Articles WHERE article_id=?", (x.article_id,)):
        raise HTTPException(404, "unknown article_id")
    with DB_LOCK:
        interaction_id = log_interaction(DB, x.user_id, x.article_id, x.feedback)
    return {"interaction_id": interaction_id, "status": "stored"}


@app.get("/articles/search")
def search(q: str, k: int = 10):
    rows = query("SELECT article_id, title, category FROM Articles WHERE title LIKE ? LIMIT ?",
                 (f"%{q}%", k))
    return [dict(zip(("article_id", "title", "category"), r)) for r in rows]


@app.get("/categories")
def categories():
    """Topics offered at onboarding: categories of the current candidate pool, most frequent first."""
    p = predictor()
    counts = defaultdict(int)
    for a in p.pool:
        if a in p.rec.category:
            counts[p.rec.category[a]] += 1
    return [{"category": c, "articles": n} for c, n in sorted(counts.items(), key=lambda x: -x[1])]


@app.get("/users/sample")
def sample_users(n: int = 8):
    """Demo readers: known readers with a history of 10 to 60 articles (else the longest), longest first."""
    hist = predictor().known_history
    by_len = sorted((u for u, h in hist.items() if len(h)), key=lambda u: (-len(hist[u]), u))
    users = [u for u in by_len if 10 <= len(hist[u]) <= 60] or by_len
    return [{"user_id": u, "n_read": len(hist[u])} for u in users[:n]]


@app.post("/users")
def new_user(body: NewUser):
    user_id = "new_" + uuid.uuid4().hex[:8]
    with DB_LOCK:
        DB.execute("INSERT INTO Users(user_id) VALUES (?)", (user_id,))
        DB.executemany("INSERT INTO UserTopics VALUES (?, ?)", [(user_id, t) for t in body.topics])
        DB.commit()
    return {"user_id": user_id, "topics": body.topics}
