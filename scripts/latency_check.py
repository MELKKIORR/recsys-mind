"""Response time of the running API, as the website sees it (SE, Week 3 Q5).

Start the API first (uvicorn api.main:app --port 8000); use 127.0.0.1, not localhost:
"localhost" costs ~2 s per call on Windows (IPv6 tried first), then:
    python scripts/latency_check.py --users 20 --rounds 2
Round 1 is usually a cold LLM cache (graph ranking served, LLM called in the background), the next rounds
are served from the cache. Also checks that a like sent to /log_interaction is stored and changes the
next recommendations. The Profile page (/user_profile, LLM summary from the cache) is timed
the same way. Writes results/latency_<date>.md and .json; exit code 1 if p95 >= 2 s.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]


def timed_get(api, path, **params):
    t0 = time.perf_counter()
    r = requests.get(f"{api}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t0) * 1000, float(r.headers.get("X-Response-Time-ms", "nan"))


def summary(values):
    v = np.array(values)
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "max": round(float(v.max()), 1),
            "share_under_2s": round(float((v < 2000).mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--wait", type=float, default=5.0, help="seconds between rounds (background LLM)")
    args = ap.parse_args()

    health = requests.get(f"{args.api}/health", timeout=60).json()
    users = [u["user_id"] for u in requests.get(f"{args.api}/users/sample",
                                                params={"n": args.users}, timeout=30).json()]
    rounds = []
    for r in range(args.rounds):
        client_ms, server_ms, status = [], [], {}
        profile_ms, summary_status = [], {}
        for u in users:
            body, c, s = timed_get(args.api, "/recommendations", user_id=u, k=args.k)
            client_ms.append(c)
            server_ms.append(s)
            status[body["llm_status"]] = status.get(body["llm_status"], 0) + 1
            prof, c, _ = timed_get(args.api, "/user_profile", user_id=u)
            profile_ms.append(c)
            st = prof.get("summary_status")
            summary_status[st] = summary_status.get(st, 0) + 1
        rounds.append({"round": r + 1, "client_ms": summary(client_ms), "server_ms": summary(server_ms),
                       "llm_status": status, "profile_client_ms": summary(profile_ms),
                       "summary_status": summary_status})
        print(f"round {r + 1}: client {rounds[-1]['client_ms']}  llm_status {status}")
        print(f"         profile {rounds[-1]['profile_client_ms']}  summary_status {summary_status}")
        if r + 1 < args.rounds:
            time.sleep(args.wait)

    # A like reaches the Interactions table and changes the next recommendations
    u = users[0]
    before, _, _ = timed_get(args.api, "/recommendations", user_id=u, k=args.k)
    liked = before["items"][0]["article_id"]
    stored = requests.post(f"{args.api}/log_interaction", timeout=30,
                           json={"user_id": u, "article_id": liked, "feedback": "like"}).json()
    hist, _, _ = timed_get(args.api, "/user_history", user_id=u)
    after, _, _ = timed_get(args.api, "/recommendations", user_id=u, k=args.k)
    like_check = {"user_id": u, "liked": liked, "interaction_id": stored.get("interaction_id"),
                  "in_database": any(x["article_id"] == liked and x["feedback"] == "like"
                                     for x in hist["reactions"]),
                  "top10_before": [i["article_id"] for i in before["items"]],
                  "top10_after": [i["article_id"] for i in after["items"]]}
    like_check["changed"] = like_check["top10_before"] != like_check["top10_after"]
    print("like check:", {k: like_check[k] for k in ("liked", "interaction_id", "in_database", "changed")})

    out = {"date": datetime.now().isoformat(timespec="seconds"), "api": args.api,
           "health": health, "users": users, "k": args.k, "rounds": rounds, "like_check": like_check,
           "server_latency": requests.get(f"{args.api}/latency", timeout=30).json()}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"latency_{stamp}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [f"# API response time ({out['date']})", "",
             f"Mode `{health['mode']}`, LLM `{health['llm']['provider']}` "
             f"(cache only: {health['llm']['cache_only']}), graph `{health['graph_version']}`, "
             f"{len(users)} readers, k={args.k}. Times in ms, measured by the client (round trip).", "",
             "| Round | p50 | p95 | max | share < 2 s | LLM status |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rounds:
        c = r["client_ms"]
        lines.append(f"| {r['round']} | {c['p50']} | {c['p95']} | {c['max']} | {c['share_under_2s']:.0%} "
                     f"| {', '.join(f'{k}: {v}' for k, v in r['llm_status'].items())} |")
    lines += ["", "Profile page (`/user_profile`, LLM summary from the cache):", "",
              "| Round | p50 | p95 | max | share < 2 s | summary status |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rounds:
        c = r["profile_client_ms"]
        lines.append(f"| {r['round']} | {c['p50']} | {c['p95']} | {c['max']} | {c['share_under_2s']:.0%} "
                     f"| {', '.join(f'{k}: {v}' for k, v in r['summary_status'].items())} |")
    lines += ["", f"Like check: reader `{u}` liked `{liked}` -> interaction #{like_check['interaction_id']}, "
              f"in the Interactions table: {like_check['in_database']}, "
              f"top 10 changed: {like_check['changed']}."]
    (ROOT / "results" / f"latency_{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved results/latency_{stamp}.md")
    worst = max(max(r["client_ms"]["p95"], r["profile_client_ms"]["p95"]) for r in rounds)
    sys.exit(0 if worst < 2000 and like_check["in_database"] else 1)


if __name__ == "__main__":
    main()
