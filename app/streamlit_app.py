
"""Streamlit frontend (SE). Talks to the API only."""

import html
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PAGES = ["Recommendations", "Profile", "History", "New reader"]

st.set_page_config(
    page_title="News Companion",
    page_icon="📰",
    layout="wide"
)

# ---------------------------------------------------------
# GLOBAL DESIGN
# ---------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ink: #18243A;
    --muted: #718096;
    --line: #E8EDF4;
    --paper: #F5F7FB;
    --white: #FFFFFF;
    --accent: #635BDB;
    --accent-soft: #F0EFFF;
    --mint: #E8F7F1;
}
.stApp {
    background:
        radial-gradient(circle at 82% 0%, rgba(222,219,255,.38), transparent 26rem),
        var(--paper);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1320px;
    padding-top: 2.1rem;
    padding-bottom: 4rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17243B 0%, #202D49 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color: #F7F8FF; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #B9C4D8;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: .25rem 0;
    font-weight: 600;
}
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: -.045em;
}
h1 {
    font-size: clamp(2.15rem, 4vw, 3.15rem) !important;
    font-weight: 800 !important;
    line-height: 1.12 !important;
}
h2 { font-size: 1.65rem !important; font-weight: 800 !important; }
h3 { font-size: 1.13rem !important; font-weight: 750 !important; letter-spacing: -.025em; }
p, li { line-height: 1.65; }
[data-testid="stCaptionContainer"] p {
    color: var(--muted);
    font-size: .82rem;
}
.cat {
    display: inline-flex;
    align-items: center;
    padding: .34rem .72rem;
    border-radius: 999px;
    background: var(--accent-soft);
    color: #554CC5;
    font-size: .68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin: .1rem 0 .55rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.94);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1rem 1.15rem;
    box-shadow: 0 8px 28px rgba(30,45,75,.045);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #D9D7FA;
    box-shadow: 0 14px 34px rgba(30,45,75,.085);
}
.stButton > button {
    border-radius: 12px;
    min-height: 2.55rem;
    border: 1px solid #E2E7F0;
    background: #FFFFFF;
    color: #34415A;
    font-weight: 700;
    transition: all .18s ease;
}
.stButton > button:hover {
    border-color: #A9A4F2;
    color: #5148C7;
    background: #F8F7FF;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #655CE0, #8278EF);
    border: 0;
    color: white;
    box-shadow: 0 7px 16px rgba(99,91,219,.22);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #554CCB, #746AE5);
    color: white;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem 1.15rem;
    box-shadow: 0 5px 18px rgba(30,45,75,.035);
}
[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-weight: 650;
}
[data-testid="stMetricValue"] {
    color: var(--ink);
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: #FBFCFE;
}
.chain {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: .4rem;
    margin: .7rem 0 .8rem;
    font-size: .78rem;
}
.chain .node {
    padding: .36rem .68rem;
    border-radius: 999px;
    border: 1px solid #E5EAF2;
    background: #F8FAFD;
    color: #53627A;
}
.chain .src { background: #EEF4FF; color: #315FA8; border-color: #DCE8FF; }
.chain .ent { background: #FFF4E9; color: #A95B18; border-color: #F8E4D0; font-weight: 700; }
.chain .dst { background: #EAF8F1; color: #217653; border-color: #D6F0E3; }
.chain .arrow { color: #A3AEC0; }
hr { border-color: var(--line); }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
}
div[data-testid="stPills"] button {
    border-radius: 999px !important;
}
@media (max-width: 700px) {
    .block-container { padding: 1.2rem 1rem 2.5rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { padding: .8rem; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# API FUNCTIONS
# ---------------------------------------------------------

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
        nodes.append(
            f'<span class="node src">📖 {html.escape(item["source_title"])}</span>'
        )

    for e in item.get("evidence_path") or []:
        nodes.append(
            f'<span class="node ent">{html.escape(e)}</span>'
        )

    nodes.append('<span class="node dst">this article</span>')

    return '<div class="chain">' + '<span class="arrow">→</span>'.join(nodes) + "</div>"


def react(user_id, item, feedback):
    body, _, _ = api(
        "POST",
        "/log_interaction",
        json={
            "user_id": user_id,
            "article_id": item["article_id"],
            "feedback": feedback
        }
    )

    verb = "Liked" if feedback == "like" else "Not interested in"

    st.session_state["last_action"] = (
        f"{verb} “{item.get('title', item['article_id'])}”: stored as "
        f"interaction #{body['interaction_id']}. Recommendations updated."
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

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

    st.title("📰 News Companion")

    st.caption("YOUR PERSONALIZED NEWS SPACE")

    st.divider()

    st.radio("Navigation", PAGES, key="page")

    st.divider()

    st.subheader("Your reader")

    st.text_input("Reader ID", key="user_id")

    sample = st.session_state.get("sample", [])

    if sample:
        st.selectbox(
            "Or pick a demo reader",
            [""] + sample,
            key="pick",
            on_change=lambda: st.session_state["pick"] and
            st.session_state.update(
                user_id=st.session_state["pick"],
                k=10
            )
        )

    st.divider()

    llm = health.get("llm") or {}

    st.caption(
        f"**Model:** {health['mode']} · graph `{health['graph_version']}`  \n"
        f"**LLM:** {llm.get('provider')} {llm.get('model') or ''} "
        f"({'cache' if llm.get('cache_only') else 'live'})"
    )

    st.divider()

    st.caption("AI-powered news discovery")


user_id = st.session_state["user_id"].strip()
page = st.session_state["page"]


# ---------------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------------

if page == "Recommendations":

    st.markdown(
        '<div style="display:flex;align-items:center;gap:.55rem;'
        'color:#655CE0;font-size:.72rem;font-weight:800;letter-spacing:.13em;'
        'text-transform:uppercase;margin-bottom:.65rem;">'
        '<span style="width:26px;height:2px;background:#8278EF;display:inline-block;"></span>'
        'YOUR PERSONALIZED NEWS FEED</div>',
        unsafe_allow_html=True
    )

    st.title("Your daily briefing")
    st.markdown(
        '<div style="max-width:690px;color:#718096;font-size:1.04rem;'
        'line-height:1.7;margin-top:-.45rem;">A smarter way to stay informed. '
        'Stories selected around your interests, reading history and feedback.</div>',
        unsafe_allow_html=True
    )

    k = st.session_state["k"]

    data, ms, server_ms = api(
        "GET",
        "/recommendations",
        params={"user_id": user_id, "k": k}
    )

    status = data["llm_status"]

    label = {
        "ok": "LLM re-ranked",
        "partial": "LLM re-ranked (partial answer)",
        "fallback_cache_miss": "graph ranking, LLM re-ranking being prepared",
        "not_used": "popular articles (no history yet)"
    }.get(status, f"graph ranking ({status})")

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,#DAD7FF,'
        'rgba(218,215,255,0));margin:1.5rem 0 1rem;"></div>',
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.caption(f"👤  Reader **{user_id}** · {label}")
    c2.caption(f"⚡  {ms:.0f} ms round trip")
    c3.caption(f"◷  Server {server_ms} ms")

    if status == "fallback_cache_miss":
        st.info(
            "The LLM re-ranking of this list is being computed "
            "in the background. Refresh in a few seconds to see it."
        )

        if st.button("↻ Refresh recommendations"):
            st.rerun()

    if msg := st.session_state.pop("last_action", None):
        st.success(msg)

    st.divider()

    st.subheader("Picked for you")
    st.caption("A selection shaped by your reading signals and interests.")

    for i, it in enumerate(data["items"], 1):

        with st.container(border=True):

            # Some article sources expose a thumbnail URL; the current
            # MIND-style API may not. In that case, show a designed fallback.
            image_url = (
                it.get("image_url")
                or it.get("thumbnail_url")
                or it.get("image")
                or it.get("thumbnail")
            )
            if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
                image_col, content_col = st.columns([1.05, 3.2], gap="large")
            else:
                image_col, content_col = None, None

            if image_col is not None:
                with image_col:
                    st.image(image_url, use_container_width=True)
                with content_col:
                    st.markdown(
                        f'<div class="cat">{html.escape(it.get("category") or "NEWS")}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"### {it.get('title') or it['article_id']}")
                    if it.get("abstract"):
                        abstract_preview = it["abstract"]
                        if len(abstract_preview) > 250:
                            abstract_preview = abstract_preview[:247].rsplit(" ", 1)[0] + "…"
                        st.markdown(
                            f'<div style="color:#66748A;font-size:.91rem;line-height:1.65;'
                            f'margin:.25rem 0 .55rem;">{html.escape(abstract_preview)}</div>',
                            unsafe_allow_html=True
                        )
                    st.caption(f"STORY {i:02d}")
            else:
                st.markdown(
                    '<div style="background:linear-gradient(115deg,#ECEBFF,#F7EFFF 55%,#E9F5F2);'
                    'border:1px solid #E9E7FA;border-radius:15px;padding:1rem 1.1rem;'
                    'margin-bottom:.9rem;display:flex;justify-content:space-between;'
                    'align-items:center;">'
                    '<span style="font-size:.72rem;font-weight:800;letter-spacing:.12em;'
                    'color:#6259C9;">THE DAILY EDIT</span>'
                    f'<span style="font-size:.75rem;color:#7D83A0;font-weight:700;">STORY {i:02d}</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="cat">{html.escape(it.get("category") or "NEWS")}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"### {it.get('title') or it['article_id']}")
                if it.get("abstract"):
                    abstract_preview = it["abstract"]
                    if len(abstract_preview) > 250:
                        abstract_preview = abstract_preview[:247].rsplit(" ", 1)[0] + "…"
                    st.markdown(
                        f'<div style="color:#66748A;font-size:.91rem;line-height:1.65;'
                        f'margin:.25rem 0 .55rem;">{html.escape(abstract_preview)}</div>',
                        unsafe_allow_html=True
                    )

            if it.get("evidence_path") or it.get("llm_reason"):
                with st.expander("✦  Why recommended"):
                    if it.get("evidence_path"):
                        st.markdown(
                            chain_html(it),
                            unsafe_allow_html=True
                        )
                    if it.get("llm_reason"):
                        st.write(it["llm_reason"])
            elif not it.get("evidence_path"):
                st.caption(it["reason"])

            if it.get("abstract"):
                with st.expander("Read full abstract"):
                    st.write(it["abstract"])

            st.markdown(
                '<div style="height:1px;background:#EDF0F5;margin:.55rem 0 .7rem;"></div>',
                unsafe_allow_html=True
            )
            buttons = st.container(horizontal=True)

            if buttons.button(
                "♡  Like",
                key=f"like_{it['article_id']}",
                width="content"
            ):
                react(user_id, it, "like")
                st.rerun()

            if buttons.button(
                "↘  Not interested",
                key=f"dislike_{it['article_id']}",
                width="content"
            ):
                react(user_id, it, "dislike")
                st.rerun()

    if st.button("Show more recommendations"):
        st.session_state["k"] = k + 10
        st.rerun()


# ---------------------------------------------------------
# PROFILE
# ---------------------------------------------------------

elif page == "Profile":

    st.caption("✦  YOUR READER PROFILE")
    st.title("Your interests")

    prof, _, _ = api(
        "GET",
        "/user_profile",
        params={"user_id": user_id}
    )

    m = st.columns(4)

    m[0].metric("Articles read", prof["n_read"])
    m[1].metric("Likes", prof["n_likes"])
    m[2].metric("Not interested", prof["n_dislikes"])
    m[3].metric("Diversity", f"{prof['diversity']:.2f}")

    if prof.get("topics"):
        st.subheader("Topics that define your feed")
        st.write(" · ".join(prof["topics"]))

    if prof.get("summary"):
        st.subheader("Your reader snapshot")
        st.info(prof["summary"])

        if prof.get("summary_status") == "fallback_cache_miss":
            st.caption(
                "Summary built from your profile numbers; "
                "the LLM summary will be ready on the next visit."
            )

    left, right = st.columns(2)

    with left:
        st.subheader("Interest by category")

        if prof["categories"]:
            st.bar_chart(
                pd.Series(prof["categories"], name="weight")
                .sort_values(ascending=False)
            )

    with right:
        st.subheader("Entities you follow")

        if prof["top_entities"]:
            st.dataframe(
                pd.DataFrame(
                    prof["top_entities"],
                    columns=["entity", "weight"]
                ),
                hide_index=True,
                use_container_width=True
            )


# ---------------------------------------------------------
# HISTORY
# ---------------------------------------------------------

elif page == "History":

    st.caption("✦  YOUR ACTIVITY")
    st.title("Reading history")

    hist, _, _ = api(
        "GET",
        "/user_history",
        params={"user_id": user_id}
    )

    st.subheader("Your feedback")

    if hist["reactions"]:
        st.dataframe(
            pd.DataFrame(hist["reactions"]),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.caption("No reaction yet.")

    st.divider()

    st.subheader("Your reading trail")

    if hist["read"]:
        st.dataframe(
            pd.DataFrame(hist["read"])[
                ["article_id", "title", "category"]
            ],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.caption("No history.")


# ---------------------------------------------------------
# NEW READER
# ---------------------------------------------------------

else:

    st.caption("✦  GET STARTED")
    st.title("Create your reader profile")

    st.write(
        "Choose at least 3 topics. Your first recommendations come "
        "from the entities of the most read articles in these topics; "
        "every like or “not interested” then refines them."
    )

    cats, _, _ = api("GET", "/categories")

    options = [c["category"] for c in cats]

    topics = st.pills(
        "Choose your interests",
        options,
        selection_mode="multi",
        key="topics"
    )

    n = len(topics or [])

    st.caption(
        f"{n} selected" +
        ("" if n >= 3 else f", {3 - n} more needed")
    )

    def create_reader():
        # callback: runs before the widgets are drawn again
        chosen = list(st.session_state["topics"])

        body, _, _ = api(
            "POST",
            "/users",
            json={"topics": chosen}
        )

        st.session_state["last_action"] = (
            f"Reader {body['user_id']} created with topics "
            f"{', '.join(chosen)}."
        )

        go("Recommendations", body["user_id"])

    st.button(
        "Create my reader",
        type="primary",
        disabled=n < 3,
        on_click=create_reader
    )
