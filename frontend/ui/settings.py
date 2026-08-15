"""Settings view: about, how it works, and a developer/debug section."""

import streamlit as st


def render_settings() -> None:
    st.markdown("## Settings")

    st.markdown("### About AgriVision")
    st.markdown(
        "AgriVision is an AI crop-health assistant. It uses a CNN model to detect "
        "crop diseases from leaf images, then explains them in farmer-friendly "
        "language with a Mistral-powered agent."
    )

    with st.expander("How it works"):
        st.markdown(
            "1. **Upload** a leaf/crop image (JPG, JPEG, or PNG).\n"
            "2. **Classify** - a CNN model predicts the disease and its confidence.\n"
            "3. **Explain** - if confidence is above 50%, a LangGraph workflow asks "
            "Mistral AI for a structured, farmer-friendly explanation.\n"
            "4. **Verify** - for low-confidence results, the app recommends manual "
            "verification by a local agricultural expert."
        )

    with st.expander("Technical details (for developers)"):
        st.markdown(
            "- Classification model: CNN (`model/crop_disease_model.keras`)\n"
            "- Class labels: 29 crop-disease classes across 10 crops\n"
            "- Workflow: LangGraph (`worker/agents/graph.py`)\n"
            "- Explanation: Mistral AI (`worker/agents/explainer_agent.py`)\n"
            "- Confidence threshold: 50%"
        )

    st.markdown("### Disclaimer")
    st.info(
        "AI results are advisory only. For severe infestations, always consult a "
        "local agricultural extension officer before applying treatments."
    )
