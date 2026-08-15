"""Main chat area: welcome, suggested prompts, image upload and messages.

Every message a user sends is answered through the real backend
(Mistral/LangGraph via frontend/backend.py) - no canned responses.
"""

import time

import streamlit as st

from frontend.backend import analyze_image, ask_question
from frontend.ui.explanation import render_explanation
from frontend.ui.result_card import render_result_card, split_prediction

SUGGESTED_PROMPTS = [
    "Analyze my crop",
    "What disease is this?",
    "How can I prevent crop diseases?",
    "Explain this disease",
]

WELCOME_HTML = """
<div style="padding: 1.2rem 0 0.4rem 0;">
  <h3 style="margin:0 0 0.35rem 0; color:#2F5233;">Welcome to AgriVision &#127793;</h3>
  <p style="margin:0 0 0.6rem 0; color:#4a4a4a;">
    Your AI assistant for crop health and disease identification.
  </p>
  <p style="margin:0 0 0.4rem 0; color:#4a4a4a;">You can:</p>
  <ul style="margin:0 0 1rem 0; color:#4a4a4a;">
    <li>Upload a crop/leaf image</li>
    <li>Ask about crop diseases</li>
    <li>Get disease explanations</li>
    <li>Understand confidence levels</li>
    <li>Ask follow-up questions</li>
  </ul>
</div>
"""


def render_chat() -> None:
    st.markdown("## Chat")

    if not st.session_state.messages:
        _render_welcome()

    for msg in st.session_state.messages:
        _render_message(msg)

    st.divider()

    uploaded = st.file_uploader(
        "Upload a leaf/crop image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="chat_uploader",
    )
    if uploaded is not None:
        file_id = getattr(uploaded, "file_id", None) or uploaded.name
        if file_id not in st.session_state.processed_files:
            st.session_state.processed_files.append(file_id)
            _handle_image(uploaded.getvalue())
            st.rerun()

    prompt = st.chat_input("Ask about crop health, or upload a leaf image to analyze...")
    if prompt:
        _handle_text(prompt)
        st.rerun()


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _render_welcome() -> None:
    st.markdown(WELCOME_HTML, unsafe_allow_html=True)
    st.markdown("**Try asking:**")
    cols = st.columns(len(SUGGESTED_PROMPTS))
    for col, prompt in zip(cols, SUGGESTED_PROMPTS):
        if col.button(prompt, width="stretch"):
            _handle_text(prompt)
            st.rerun()


def _render_message(msg: dict) -> None:
    is_user = msg["role"] == "user"
    with st.chat_message(
        "user" if is_user else "assistant",
        avatar="🧑‍🌾" if is_user else "🌱",
    ):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("image") is not None:
            st.image(msg["image"], caption="Leaf Image", width=260)
        if msg.get("result"):
            render_result_card(msg["result"])
            if msg["result"].get("explanation"):
                render_explanation(msg["result"]["explanation"])


# --------------------------------------------------------------------------- #
# Message handling (all responses come from the real backend)
# --------------------------------------------------------------------------- #


def _handle_image(image_bytes: bytes) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": "Uploaded a leaf/crop image", "image": image_bytes, "result": None}
    )
    st.session_state.pending_image = None

    with st.spinner("Analyzing image..."):
        try:
            result = analyze_image(image_bytes)
        except Exception as exc:  # noqa: BLE001 - surface errors to the farmer
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Analysis failed: {exc}", "image": None, "result": None}
            )
            return

    prediction = result.get("predicted_disease") or "Unknown"
    confidence = result.get("confidence") or 0.0
    explanation = result.get("explanation")
    crop, disease = split_prediction(prediction)

    st.session_state.analyses.append(
        {
            "crop": crop,
            "disease": disease,
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation,
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        }
    )

    st.session_state.disease_context = {
        "disease": prediction,
        "confidence": confidence,
        "explanation": explanation,
    }

    st.session_state.messages.append(
        {"role": "assistant", "content": None, "image": None, "result": result}
    )


def _handle_text(text: str) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": text, "image": None, "result": None}
    )

    context = st.session_state.disease_context
    with st.spinner("Thinking..."):
        try:
            reply = ask_question(
                text,
                disease=context["disease"] if context else None,
                confidence=context["confidence"] if context else None,
            )
        except Exception as exc:  # noqa: BLE001 - surface errors to the farmer
            reply = f"Sorry, I could not get a response from the AI assistant. ({exc})"

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "image": None, "result": None}
    )
