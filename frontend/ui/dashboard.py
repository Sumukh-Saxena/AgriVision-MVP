"""Farm Dashboard - high-level overview built from real session data.

No fake statistics: numbers are computed from st.session_state.analyses.
"""

import streamlit as st


def render_dashboard() -> None:
    st.markdown("## Farm Dashboard")
    st.markdown("### Farm Overview")

    analyses = st.session_state.analyses
    if not analyses:
        st.info("No analyses yet. Upload a crop image in the Chat view to get started.")
        return

    total = len(analyses)
    confidences = [a["confidence"] for a in analyses if a.get("confidence") is not None]
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    diseases = {a["disease"] for a in analyses if a.get("disease") != "Healthy"}

    c1, c2, c3 = st.columns(3)
    c1.metric("Crop Checks", total)
    c2.metric("Avg. Confidence", f"{avg_conf * 100:.0f}%")
    c3.metric("Diseases Detected", len(diseases))

    st.divider()

    st.markdown("### Recent Analyses")
    rows = [
        {
            "Crop": a["crop"],
            "Disease": a["disease"],
            "Confidence": f"{a['confidence'] * 100:.0f}%" if a.get("confidence") is not None else "—",
            "Date": a.get("timestamp", ""),
        }
        for a in reversed(analyses)
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
