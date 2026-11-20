"""Streamlit frontend (SE). Talks to the API only.  Run: streamlit run app/streamlit_app.py

API_URL defaults to http://127.0.0.1:8000 (on Windows "localhost" adds ~2 s per call: IPv6 tried first).
"""
import html
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PAGES = ["Recommendations", "Profile", "History", "New reader"]
st.set_page_config(page_title="News companion", page_icon="📰", layout="wide")

st.markdown("""<style>
.chain {display:flex; flex-wrap:wrap; align-items:center; gap:.35rem; margin:.2rem 0 .4rem 0;
        font-size:.9rem}
.chain .node {padding:.15rem .55rem; border-radius:1rem; border:1px solid rgba(128,128,128,.35)}
.chain .src {background:rgba(70,130,180,.12)}
.chain .ent {background:rgba(255,165,0,.15); font-weight:600}
.chain .dst {background:rgba(60,179,113,.15)}
.chain .arrow {opacity:.6}
.cat {font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; opacity:.7}
</style>""", unsafe_allow_html=True)


def api(method, path, **kw):
    """Call the API; returns (json, round-trip ms, server ms)."""
    t0 = time.perf_counter()
    r = requests.request(method, f"{API}{path}", timeout=30, **kw)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), ms, r.headers.get("X-Response-Time-ms")


def go(page, user_id=None):
    st.session_state["page"] = page
    if user_id:
        st.session_state["user_id"] = user_id
    st.session_state["k"] = 10


def chain_html(item):
    """Evidence path as a chain: article read -> entity -> linked entity -> recommended article."""
    nodes = []
    if item.get("source_title"):
        nodes.append(f'<span class="node src">📖 {html.escape(item["source_title"])}</span>')
    for e in item.get("evidence_path") or []:
        nodes.append(f'<span class="node ent">{html.escape(e)}</span>')
    nodes.append('<span class="node dst">this article</span>')
    return '<div class="chain">' + '<span class="arrow">→</span>'.join(nodes) + "</div>"


def react(user_id, item, feedback):
    body, _, _ = api("POST", "/log_interaction",
                     json={"user_id": user_id, "article_id": item["article_id"], "feedback": feedback})
    verb = "Liked" if feedback == "like" else "Not interested in"
    st.session_state["last_action"] = (f"{verb} “{item.get('title', item['article_id'])}”: stored as "
                                       f"interaction #{body['interaction_id']}. Recommendations updated.")


# Sidebar -------------------------------------------------------------------------------------------
try:
    health, _, _ = api("GET", "/health")
except requests.RequestException as e:
    st.error(f"The API at {API} does not answer: {e}")
    st.stop()

st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("k", 10)
if "user_id" not in st.session_state:
    sample = api("GET", "/users/sample", params={"n": 8})[0]
    st.session_state["sample"] = [u["user_id"] for u in sample]
    st.session_state["user_id"] = sample[0]["user_id"] if sample else "U13740"

with st.sidebar:
    st.title("📰 News companion")
    st.radio("Page", PAGES, key="page")
    st.divider()
    st.text_input("Reader id", key="user_id")
    sample = st.session_state.get("sample", [])
    if sample:
        st.selectbox("Or pick a demo reader", [""] + sample, key="pick",
                     on_change=lambda: st.session_state["pick"] and
                     st.session_state.update(user_id=st.session_state["pick"], k=10))
    st.divider()
    llm = health.get("llm") or {}
    st.caption(f"Model: **{health['mode']}** · graph `{health['graph_version']}`  \n"
               f"LLM re-ranking: **{llm.get('provider')}** {llm.get('model') or ''} "
               f"({'served from cache' if llm.get('cache_only') else 'live'})")

user_id = st.session_state["user_id"].strip()
page = st.session_state["page"]

# Recommendations -----------------------------------------------------------------------------------
if page == "Recommendations":
    st.header("Recommendations")
    k = st.session_state["k"]
    data, ms, server_ms = api("GET", "/recommendations", params={"user_id": user_id, "k": k})
    status = data["llm_status"]
    label = {"ok": "LLM re-ranked", "partial": "LLM re-ranked (partial answer)",
             "fallback_cache_miss": "graph ranking, LLM re-ranking being prepared",
             "not_used": "popular articles (no history yet)"}.get(status, f"graph ranking ({status})")
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.caption(f"Reader **{user_id}** · {label}")
    c2.caption(f"⏱ {ms:.0f} ms round trip")
    c3.caption(f"server {server_ms} ms")
    if status == "fallback_cache_miss":
        st.info("The LLM re-ranking of this list is being computed in the background. "
                "Refresh in a few seconds to see it.")
        st.button("Refresh")
    if msg := st.session_state.pop("last_action", None):
        st.success(msg)

    for i, it in enumerate(data["items"], 1):
        with st.container(border=True):
            st.markdown(f'<div class="cat">{html.escape(it.get("category") or "")}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"**{i}. {it.get('title') or it['article_id']}**")
            if it.get("evidence_path"):
                st.markdown(chain_html(it), unsafe_allow_html=True)
            else:
                st.caption(it["reason"])
            if it.get("llm_reason"):
                st.caption(f"🤖 {it['llm_reason']}")
            if it.get("abstract"):
                with st.expander("Abstract"):
                    st.write(it["abstract"])
            buttons = st.container(horizontal=True)
            if buttons.button("👍 Like", key=f"like_{it['article_id']}", width="content"):
                react(user_id, it, "like")
                st.rerun()
            if buttons.button("👎 Not interested", key=f"dislike_{it['article_id']}", width="content"):
                react(user_id, it, "dislike")
                st.rerun()
    if st.button("Show more"):
        st.session_state["k"] = k + 10
        st.rerun()

# Profile -------------------------------------------------------------------------------------------
elif page == "Profile":
    st.header("Profile")
    prof, _, _ = api("GET", "/user_profile", params={"user_id": user_id})
    m = st.columns(4)
    m[0].metric("Articles read", prof["n_read"])
    m[1].metric("Likes", prof["n_likes"])
    m[2].metric("Not interested", prof["n_dislikes"])
    m[3].metric("Diversity", f"{prof['diversity']:.2f}")
    if prof.get("topics"):
        st.write("Topics chosen at onboarding: " + ", ".join(prof["topics"]))
    if prof.get("summary"):
        st.info(prof["summary"])
    left, right = st.columns(2)
    with left:
        st.subheader("Categories")
        if prof["categories"]:
            st.bar_chart(pd.Series(prof["categories"], name="weight").sort_values(ascending=False))
    with right:
        st.subheader("Main entities")
        if prof["top_entities"]:
            st.dataframe(pd.DataFrame(prof["top_entities"], columns=["entity", "weight"]),
                         hide_index=True, use_container_width=True)

# History -------------------------------------------------------------------------------------------
elif page == "History":
    st.header("History")
    hist, _, _ = api("GET", "/user_history", params={"user_id": user_id})
    st.subheader("Reactions on this site")
    if hist["reactions"]:
        st.dataframe(pd.DataFrame(hist["reactions"]), hide_index=True, use_container_width=True)
    else:
        st.caption("No reaction yet.")
    st.subheader("Articles read (MIND history)")
    if hist["read"]:
        st.dataframe(pd.DataFrame(hist["read"])[["article_id", "title", "category"]],
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No history.")

# Onboarding ----------------------------------------------------------------------------------------
else:
    st.header("New reader")
    st.write("Choose at least 3 topics. Your first recommendations come from the entities of the most "
             "read articles in these topics; every like or “not interested” then refines them.")
    cats, _, _ = api("GET", "/categories")
    options = [c["category"] for c in cats]
    topics = st.pills("Topics", options, selection_mode="multi", key="topics")
    n = len(topics or [])
    st.caption(f"{n} selected" + ("" if n >= 3 else f", {3 - n} more needed"))

    def create_reader():
        # callback: runs before the widgets are drawn again, so it may change the page and reader id
        chosen = list(st.session_state["topics"])
        body, _, _ = api("POST", "/users", json={"topics": chosen})
        st.session_state["last_action"] = f"Reader {body['user_id']} created with topics {', '.join(chosen)}."
        go("Recommendations", body["user_id"])

    st.button("Create my reader", type="primary", disabled=n < 3, on_click=create_reader)
