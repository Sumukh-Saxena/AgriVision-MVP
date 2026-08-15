"""Left sidebar: branding, navigation, new-chat and recent conversations."""

import streamlit as st

from frontend.state import PAGES, new_chat


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="av-brand">&#127793; AgriVision</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="av-brand-sub">AI Farming Assistant</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        if st.button("+ New Chat", width="stretch", type="primary"):
            new_chat()
            st.session_state.page = "Chat"
            st.session_state["nav"] = "Chat"
            st.rerun()

        st.divider()

        page = st.radio(
            "Navigation",
            PAGES,
            key="nav",
            index=PAGES.index(st.session_state.page),
            label_visibility="collapsed",
        )
        st.session_state.page = page

        st.markdown(
            '<div class="av-section-title">Recent Chats</div>',
            unsafe_allow_html=True,
        )
        _render_recent_chats()


def _render_recent_chats() -> None:
    analyses = st.session_state.analyses
    if not analyses:
        st.caption("No conversations yet.")
        return

    for i, item in enumerate(reversed(analyses[-5:])):
        idx = len(analyses) - 1 - i
        label = f"{item['crop']} – {item['disease']}"
        if st.button(label, key=f"recent_{i}", width="stretch"):
            st.session_state.page = "History"
            st.session_state["nav"] = "History"
            st.session_state.expanded_analysis = idx
            st.rerun()
