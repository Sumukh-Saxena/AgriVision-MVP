"""Clean prediction result card shown after image classification."""

import streamlit as st

CONFIDENCE_THRESHOLD = 0.50  # matches worker/agents/graph.py


def _confidence_label(confidence: float) -> str:
    if confidence > 0.70:
        return "High confidence"
    if confidence > CONFIDENCE_THRESHOLD:
        return "Moderate confidence"
    return "Low confidence — manual verification recommended"


def split_prediction(prediction: str) -> tuple[str, str]:
    """Split a 'Crop - Disease' label into (crop, disease)."""
    if " - " in prediction:
        crop, disease = prediction.split(" - ", 1)
        return crop, disease
    return prediction, prediction


def render_result_card(result: dict) -> None:
    """Render the Crop Diagnosis card for a workflow result dict.

    Expected keys: predicted_disease, confidence, explanation.
    """
    prediction = result.get("predicted_disease") or "Unknown"
    confidence = result.get("confidence") or 0.0
    explanation = result.get("explanation")

    crop, disease = split_prediction(prediction)
    pct = confidence * 100

    with st.container(border=True):
        st.markdown("##### Crop Diagnosis")

        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"**{crop}**")
            st.markdown(disease)
        with right:
            st.markdown("**Confidence**")
            st.markdown(f"**{pct:.0f}%**")

        st.progress(min(confidence, 1.0))
        st.caption(_confidence_label(confidence))

        if explanation:
            st.caption("AI explanation available &#10003;")
        else:
            st.caption("AI explanation not generated (low confidence)")

        if confidence <= CONFIDENCE_THRESHOLD:
            st.warning(
                "The model is not sufficiently confident about this prediction. "
                "Please verify the image or consult a local agricultural expert."
            )
