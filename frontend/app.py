"""AgriVision - AI Crop Health Assistant.

Transforms the basic classifier into a chat + dashboard interface while keeping
the existing CNN / LangGraph / Mistral backend as the single source of truth.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from frontend.state import PAGES, init_state
from frontend.ui.chat import render_chat
from frontend.ui.dashboard import render_dashboard
from frontend.ui.history import render_history
from frontend.ui.settings import render_settings
from frontend.ui.sidebar import render_sidebar
from frontend.ui.styles import apply as apply_styles

st.set_page_config(
    page_title="AgriVision AI Farming Assistant",
    page_icon="🌱",
    layout="wide",
)

apply_styles()
init_state()
render_sidebar()

if st.session_state.page == "Chat":
    render_chat()
elif st.session_state.page == "Farm Dashboard":
    render_dashboard()
elif st.session_state.page == "History":
    render_history()
elif st.session_state.page == "Settings":
    render_settings()
