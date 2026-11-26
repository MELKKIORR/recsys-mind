"""Streamlit frontend (SE). Talks to the API only."""

import html
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PAGES = ["Recommendations", "Profile", "History", "New reader"]

st.set_page_config(page_title="News Companion", page_icon="📰", layout="wide")

# ---------------------------------------------------------
# GLOBAL DESIGN  ("Evening edition": forest ink, saffron marker, cool paper)
# ---------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=Instrument+Sans:wght@400;500;600;700&display=swap');

:root {
    --paper: #ECEEE9;
    --card: #F8F9F6;
    --ink: #10231C;
    --forest: #0E2B22;
    --forest-2: #17392E;
    --saffron: #F5B800;
    --saffron-soft: #FBEFC2;
    --muted: #5B6761;
    --line: #CBD2CB;
}

.stApp { background: var(--paper); color: var(--ink); font-family: 'Instrument Sans', sans-serif; }
.block-container { max-width: 1100px; padding-top: 2.5rem; padding-bottom: 5rem; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Typography ---------- */
h1, h2, h3 { color: var(--ink) !important; font-family: 'Fraunces', Georgia, serif !important; font-variation-settings: "opsz" 144; }
h1 { font-size: clamp(3rem, 6vw, 5.2rem) !important; font-weight: 800 !important; letter-spacing: -.035em; line-height: .98 !important; padding: 0 0 .4rem !important; }
h2 { font-size: 2rem !important; font-weight: 600 !important; letter-spacing: -.02em; }
h3 { font-size: 1.55rem !important; font-weight: 600 !important; line-height: 1.2 !important; letter-spacing: -.01em; }
p, li { line-height: 1.7; }
[data-testid="stCaptionContainer"] p { color: var(--muted); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--forest) !important; border-right: 0; }
[data-testid="stSidebar"] * { color: #E9EFE9 !important; }
[data-testid="stSidebar"] h1 { font-size: 2.1rem !important; letter-spacing: -.03em; line-height: 1.05 !important; color: #fff !important; }
[data-testid="stSidebar"] h3 { font-size: 1.25rem !important; color: var(--saffron) !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #9DB3A8 !important; font-size: .82rem; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .35rem .6rem; border-radius: 6px; transition: background .15s; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.07); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: var(--saffron); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * { color: var(--forest) !important; font-weight: 700; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--forest-2) !important; border: 1px solid rgba(255,255,255,.18) !important; color: #fff !important; border-radius: 6px;
}
[data-testid="stSidebar"] input:focus { border-color: var(--saffron) !important; }

/* ---------- Eyebrow / hero pieces ---------- */
.eyebrow { display: flex; align-items: center; gap: .6rem; color: var(--muted); font-size: .85rem; font-weight: 600; margin-bottom: .9rem; }
.eyebrow i { width: 10px; height: 10px; border-radius: 50%; background: var(--saffron); box-shadow: 0 0 0 4px var(--saffron-soft); }
.lede { max-width: 560px; color: var(--muted); font-size: 1.12rem; line-height: 1.6; margin: .2rem 0 1.6rem; }

/* Status strip (replaces the three stat boxes) */
.strip { display: flex; flex-wrap: wrap; gap: .4rem 1.6rem; padding: .7rem 0; border-top: 2px solid var(--ink); border-bottom: 1px solid var(--line); font-size: .86rem; color: var(--muted); }
.strip b { color: var(--ink); font-weight: 700; }
.strip span.live::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--saffron); margin-right: .45rem; }

/* Alerts */
[data-testid="stAlert"] { background: var(--saffron-soft); border: 0; border-left: 4px solid var(--saffron); border-radius: 0; color: var(--ink); }
[data-testid="stAlert"] * { color: var(--ink) !important; }

/* ---------- Article cards ---------- */
[class*="st-key-card_"] { background: var(--card); border: 1px solid var(--line); border-radius: 4px; padding: 1.6rem 1.8rem; box-shadow: none; transition: border-color .2s, box-shadow .2s; }
[class*="st-key-card_"]:hover { border-color: var(--ink); box-shadow: 6px 6px 0 var(--saffron); }
.stButton > button { white-space: nowrap; }
[data-testid="stImage"] img { border-radius: 2px; object-fit: cover; }

.rank { display: flex; align-items: baseline; gap: 1rem; margin-bottom: .6rem; }
.rank .n { font-family: 'Fraunces', serif; font-weight: 800; font-size: 3.4rem; line-height: .9; letter-spacing: -.04em; color: var(--saffron); -webkit-text-stroke: 1.5px var(--ink); }
.rank .c { font-size: .8rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--ink); border-bottom: 2px solid var(--saffron); padding-bottom: 2px; }
.cat { display: inline-block; font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; border-bottom: 2px solid var(--saffron); padding-bottom: 2px; margin: .4rem 0; }
.abs { color: var(--muted); font-size: .98rem; line-height: 1.65; margin: .3rem 0 .8rem; max-width: 68ch; }

/* Evidence chain */
.chain { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; margin: .4rem 0; }
.chain .node { padding: .3rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600; border: 1px solid var(--line); background: #fff; }
.chain .src { background: var(--forest); color: #fff; border-color: var(--forest); }
.chain .ent { background: var(--saffron-soft); border-color: #E6CF7A; }
.chain .dst { background: var(--saffron); border-color: var(--saffron); }
.chain .arrow { color: var(--muted); }

/* ---------- Controls ---------- */
.stButton > button { border-radius: 4px; border: 1.5px solid var(--ink); background: transparent; color: var(--ink); font-weight: 600; transition: background .15s, color .15s, transform .1s; }
.stButton > button:hover { background: var(--saffron); border-color: var(--ink); color: var(--ink); }
.stButton > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"] { background: var(--ink); color: #fff; }
.stButton > button[kind="primary"]:hover { background: var(--saffron); color: var(--ink); }
.stButton > button:focus-visible, input:focus-visible { outline: 3px solid var(--saffron); outline-offset: 2px; }

[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-top: 3px solid var(--ink); border-radius: 0; padding: 1rem 1.1rem; }
[data-testid="stMetricValue"] { font-family: 'Fraunces', serif; font-weight: 800; font-size: 2.4rem; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 4px; background: transparent; }
hr { border-color: var(--line); }

/* One page-load reveal, nothing else moves on its own */
.block-container > div > [data-testid="stVerticalBlock"] > div:first-child { animation: rise .5s ease both; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
@media (max-width: 700px) { .block-container { padding: 1rem; } .rank .n { font-size: 2.6rem; } }
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


def abstract_html(text):
    """Truncated abstract preview (design helper)."""
    if len(text) > 250:
        text = text[:247].rsplit(" ", 1)[0] + "…"
    return f'<div class="abs">{html.escape(text)}</div>'


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

    st.caption("Your personalized news space")

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
        '<div class="eyebrow"><i></i>Your personalized news feed</div>',
        unsafe_allow_html=True
    )

    st.title("Your daily briefing")
    st.markdown(
        '<div class="lede">A smarter way to stay informed. '
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
        f'<div class="strip">'
        f'<span class="live">Reader <b>{html.escape(user_id)}</b> · {html.escape(label)}</span>'
        f'<span>{ms:.0f} ms round trip</span>'
        f'<span>Server {server_ms} ms</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if status == "fallback_cache_miss":
        st.info(
            "The LLM re-ranking of this list is being computed "
            "in the background. Refresh in a few seconds to see it."
        )

        if st.button("↻ Refresh recommendations", type="primary"):
            st.rerun()

    if msg := st.session_state.pop("last_action", None):
        st.success(msg)

    st.divider()

    st.subheader("Picked for you")
    st.caption("A selection shaped by your reading signals and interests.")

    for i, it in enumerate(data["items"], 1):

        with st.container(border=True, key=f"card_{i}"):

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

            cat = html.escape(it.get("category") or "NEWS")

            if image_col is not None:
                with image_col:
                    st.image(image_url, use_container_width=True)
                with content_col:
                    st.markdown(
                        f'<div class="rank"><span class="n">{i:02d}</span>'
                        f'<span class="c">{cat}</span></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"### {it.get('title') or it['article_id']}")
                    if it.get("abstract"):
                        st.markdown(abstract_html(it["abstract"]), unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="rank"><span class="n">{i:02d}</span>'
                    f'<span class="c">{cat}</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"### {it.get('title') or it['article_id']}")
                if it.get("abstract"):
                    st.markdown(abstract_html(it["abstract"]), unsafe_allow_html=True)

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

            button_col1, button_col2, _spacer = st.columns([1, 1.5, 3])

            if button_col1.button(
                "♡  Like",
                key=f"like_{it['article_id']}"
            ):
                react(user_id, it, "like")
                st.rerun()

            if button_col2.button(
                "↘  Not interested",
                key=f"dislike_{it['article_id']}"
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

    st.markdown('<div class="eyebrow"><i></i>Your reader profile</div>', unsafe_allow_html=True)
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
                .sort_values(ascending=False),
                color="#0E2B22"
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

    st.markdown('<div class="eyebrow"><i></i>Your activity</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="eyebrow"><i></i>Get started</div>', unsafe_allow_html=True)
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