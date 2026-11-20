"""LLM profile summary of a reader (MLE-Ops, Week 2 task caught up in Week 3).

The LLM gets the structured profile (top categories and entities, ignored topics, diversity) and the
titles of the last clicked articles, and writes two short sentences about the reader. The summary is
shown on the Profile page of the site and in the report. It is not used to rank.

Same rules as the re-ranking: same client, same disk cache, JSON answer checked by code. If the answer
is invalid, too long, or the call fails, a summary written from the profile numbers is used instead,
so the site always has a text.

    python -m src.llm.profile_summary [config] --users 20     # precompute summaries for the demo
"""
import argparse
import json

import pandas as pd

from src.config import ROOT, load_config
from src.llm.rerank import describe_profile, extract_json

SYSTEM = (
    "You describe the reading interests of one reader of a news site, for the reader to see on their "
    "profile page. Use only the profile and the titles you are given, do not guess anything personal "
    "(age, job, place, politics, health). Write two short sentences in plain English, second person "
    '("You mostly read..."). Answer with JSON only: {"summary": "<two sentences>"}')
MAX_CHARS = 400


def template_summary(desc: dict) -> str:
    """Summary built from the numbers only: used when the LLM answer is missing or invalid."""
    cats = list(desc["top_categories"])[:3]
    ents = desc["top_entities"][:3]
    if not cats:
        return "We do not know your interests yet: like a few articles or pick three topics."
    text = "You mostly read " + ", ".join(cats)
    text += f", often about {', '.join(ents)}." if ents else "."
    if desc["ignored_categories"]:
        text += f" You tend to skip {', '.join(desc['ignored_categories'][:2])}."
    return text


def summarize(rec, profile, client, n_titles=10) -> dict:
    desc = describe_profile(rec, profile)
    titles = [rec.title.get(a, "") for a in profile.history[-n_titles:] if rec.title.get(a)]
    if not titles and not desc["top_categories"]:
        return {"summary": template_summary(desc), "status": "empty_profile"}
    payload = {"task": "profile_summary", "profile": desc, "recent_titles": titles}
    try:
        text = client.complete(SYSTEM, json.dumps(payload, ensure_ascii=False))
    except RuntimeError:
        return {"summary": template_summary(desc), "status": "fallback_llm_error"}
    data = extract_json(text)
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(summary, str) or not summary.strip() or len(summary) > MAX_CHARS:
        return {"summary": template_summary(desc), "status": "fallback_invalid"}
    return {"summary": summary.strip(), "status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--users", type=int, default=20, help="number of readers (longest histories first)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from src.graph.retrieval import GraphRecommender
    from src.llm.client import LLMClient
    rec = GraphRecommender.from_config(cfg)
    client = LLMClient.from_config(cfg)
    beh = pd.concat([pd.read_pickle(cfg["processed_dir"] / f"behaviors_{s}.pkl") for s in ("train", "val")])
    last = beh.sort_values("time").groupby("user_id")["history"].last()
    users = last.map(len).sort_values(ascending=False).index[:args.users]
    out, statuses = {}, {}
    for u in users:
        profile = rec.build_profile(list(last[u]), rec.ignored.get(u, []))
        res = summarize(rec, profile, client)
        out[u] = res
        statuses[res["status"]] = statuses.get(res["status"], 0) + 1
    path = ROOT / cfg.get("results_dir", "results") / "profile_summaries.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"provider": client.provider, "model": client.model, "statuses": statuses,
                                "client": client.stats, "summaries": out}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"{len(out)} summaries, statuses {statuses}, LLM {client.stats}. Saved {path}")


if __name__ == "__main__":
    main()
