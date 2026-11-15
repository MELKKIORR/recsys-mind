"""Streamlit frontend (SE). Talks to the API only.  Run: streamlit run app/streamlit_app.py"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="News companion", layout="wide")

user_id = st.sidebar.text_input("Reader id", value="U13740")
page = st.sidebar.radio("Page", ["Recommendations", "Profile", "History"])

if page == "Recommendations":
    st.header("Recommendations")
    k = st.session_state.setdefault("k", 10)
    items = requests.get(f"{API}/recommendations", params={"user_id": user_id, "k": k}).json()["items"]
    for i, it in enumerate(items, 1):
        st.subheader(f"{i}. {it.get('title', it['article_id'])}")
        st.caption(it.get("category", ""))
        st.write(it["reason"])
        c1, c2, _ = st.columns([1, 2, 6])
        for col, label, fb in ((c1, "Like", "like"), (c2, "Not interested", "dislike")):
            if col.button(label, key=f"{fb}_{it['article_id']}"):
                requests.post(f"{API}/log_interaction",
                              json={"user_id": user_id, "article_id": it["article_id"], "feedback": fb})
                st.rerun()
    if st.button("Show more"):
        st.session_state["k"] = k + 10
        st.rerun()
elif page == "Profile":
    st.header("Profile")
    st.json(requests.get(f"{API}/user_profile", params={"user_id": user_id}).json())
    topics = st.multiselect("New reader: choose at least 3 topics",
                            ["news", "sports", "finance", "autos", "lifestyle", "health", "travel"])
    if st.button("Create reader") and len(topics) >= 3:
        st.success(requests.post(f"{API}/users", json={"topics": topics}).json())
else:
    st.header("History")
    st.json(requests.get(f"{API}/user_history", params={"user_id": user_id}).json())
