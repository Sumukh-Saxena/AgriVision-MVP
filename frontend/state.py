"""Session-state management for the AgriVision app.

Everything is kept in Streamlit's session state so no database is required.
If persistent storage is needed later, this is the single place to swap in a
backend (e.g. SQLite) without touching the UI modules.
"""

import streamlit as st

DEFAULT_PAGE = "Chat"
PAGES = ["Chat", "Farm Dashboard", "History", "Settings"]


def init_state() -> None:
    """Initialise every session-state slot used by the app."""
    st.session_state.setdefault("page", DEFAULT_PAGE)
    st.session_state.setdefault("nav", DEFAULT_PAGE)
    # Chat conversation: list of {"role", "content", "image", "result"}
    st.session_state.setdefault("messages", [])
    # Completed analyses: list of {"crop", "disease", "prediction",
    # "confidence", "explanation", "timestamp"}
    st.session_state.setdefault("analyses", [])
    # Context of the most recent analysis, used to ground follow-up questions.
    st.session_state.setdefault("disease_context", None)
    # Uploaded-but-unprocessed image bytes.
    st.session_state.setdefault("pending_image", None)
    # File ids already analysed (avoids re-analysing on every rerun).
    st.session_state.setdefault("processed_files", [])
    # History item whose details are currently expanded.
    st.session_state.setdefault("expanded_analysis", None)


def new_chat() -> None:
    """Clear the current conversation (keeps session analyses for dashboard)."""
    st.session_state.messages = []
    st.session_state.disease_context = None
    st.session_state.pending_image = None
