"""Analysis History - session-based list of previous image analyses."""

import streamlit as st

from frontend.ui.explanation import render_explanation
from frontend.ui.result_card import render_result_card
from frontend.ui.weather import render_weather


def render_history() -> None:
    st.markdown("## Analysis History")

    analyses = st.session_state.analyses
    if not analyses:
        st.info("No analyses yet. Upload a crop image in the Chat view to get started.")
        return

    for i, a in enumerate(reversed(analyses)):
        idx = len(analyses) - 1 - i
        with st.container(border=True):
            cols = st.columns([2, 3, 2, 2, 1])
            cols[0].markdown(f"**{a['crop']}**")
            cols[1].markdown(a["disease"])
            cols[2].markdown(
                f"{a['confidence'] * 100:.0f}%" if a.get("confidence") is not None else "—"
            )
            cols[3].markdown(a.get("timestamp", ""))
            if cols[4].button("View", key=f"view_{idx}", width="stretch"):
                st.session_state.expanded_analysis = idx

    expanded = st.session_state.expanded_analysis
    if expanded is not None and 0 <= expanded < len(analyses):
        a = analyses[expanded]
        st.divider()
        if a.get("location"):
            st.markdown(f"**Location:** {a['location']}")
        render_result_card(
            {
                "predicted_disease": a.get("prediction") or a.get("disease"),
                "confidence": a.get("confidence"),
                "explanation": a.get("explanation"),
            }
        )
        if a.get("weather"):
            render_weather(a["weather"])
        if a.get("explanation"):
            render_explanation(a["explanation"])
