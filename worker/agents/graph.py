from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from worker.agents.explainer_agent import DiseaseExplainerAgent
from worker.model import model as model_api

CONFIDENCE_THRESHOLD = 0.50  # proceed to LLM pipeline only if confidence > 50%


class CropDiseaseState(TypedDict):
    image_path: str
    predicted_disease: Optional[str]
    confidence: Optional[float]
    explanation: Optional[str]
    message: Optional[str]


class CropDiseaseWorkflow:
    """LangGraph pipeline: CNN classify -> confidence gate -> Mistral explain."""

    def __init__(
        self,
        explainer_agent: DiseaseExplainerAgent,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.explainer = explainer_agent
        self.threshold = threshold
        self.graph = self._build_graph()

    def _classify(self, state: CropDiseaseState) -> CropDiseaseState:
        # Image is resized to 256x256 and fed to the CNN model inside `model_api.predict`.
        predicted_disease, confidence = model_api.predict(state["image_path"])
        return {
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "explanation": None,
            "message": None,
        }

    def _explain(self, state: CropDiseaseState) -> CropDiseaseState:
        explanation = self.explainer.explain(
            disease_name=state["predicted_disease"],
            confidence=state["confidence"],
        )
        return {"explanation": explanation, "message": "Explanation generated."}

    def _route(self, state: CropDiseaseState) -> str:
        confidence = state.get("confidence") or 0.0
        if confidence > self.threshold:
            return "explain"
        return "end"

    def _build_graph(self):
        graph = StateGraph(CropDiseaseState)

        graph.add_node("classify", self._classify)
        graph.add_node("explain", self._explain)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify",
            self._route,
            {"explain": "explain", "end": END},
        )
        graph.add_edge("explain", END)

        return graph.compile()

    def run(self, image_path: str) -> CropDiseaseState:
        """Run the full pipeline on an image and return the final state."""
        result = self.graph.invoke({"image_path": image_path, "predicted_disease": None,
                                    "confidence": None, "explanation": None, "message": None})
        return result
