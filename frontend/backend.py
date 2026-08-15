"""Thin wrappers around the existing AgriVision backend.

These only call the existing model / LangGraph / Mistral components.
No model logic is reimplemented here - the backend remains the source of truth.
"""

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st


@st.cache_resource
def get_workflow():
    """Return the cached LangGraph workflow (CNN classify -> route -> explain)."""
    from worker.agents.explainer_agent import DiseaseExplainerAgent
    from worker.agents.graph import CropDiseaseWorkflow

    explainer = DiseaseExplainerAgent()
    return CropDiseaseWorkflow(explainer_agent=explainer)


def analyze_image(image_bytes: bytes) -> dict:
    """Run the existing workflow on an uploaded image.

    Returns the workflow state: predicted_disease, confidence and (only when
    confidence > 50%) the Mistral-generated explanation.
    """
    workflow = get_workflow()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        state = workflow.run(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "predicted_disease": state.get("predicted_disease"),
        "confidence": state.get("confidence"),
        "explanation": state.get("explanation"),
    }


def ask_question(
    question: str,
    disease: Optional[str] = None,
    confidence: Optional[float] = None,
) -> str:
    """Route a follow-up question through the existing Mistral agent.

    When a recent analysis exists, its disease/confidence is passed as context
    so the answer stays grounded in the detected condition.
    """
    workflow = get_workflow()
    return workflow.explainer.chat(
        question,
        disease_name=disease,
        confidence=confidence,
    )
