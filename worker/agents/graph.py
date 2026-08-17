from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from worker.agents.explainer_agent import DiseaseExplainerAgent
from worker.model import model as model_api
from worker.weather import WeatherService

CONFIDENCE_THRESHOLD = 0.50  # proceed to LLM pipeline only if confidence > 50%


class CropDiseaseState(TypedDict):
    image_path: str
    location: Optional[str]
    predicted_disease: Optional[str]
    confidence: Optional[float]
    explanation: Optional[str]
    weather: Optional[dict]
    regional: Optional[str]
    message: Optional[str]


class CropDiseaseWorkflow:
    """LangGraph pipeline: CNN classify -> weather -> confidence gate -> Mistral explain.

    Weather is fetched for the farmer-provided location (when available) without
    breaking the core classification/explanation flow.
    """

    def __init__(
        self,
        explainer_agent: DiseaseExplainerAgent,
        threshold: float = CONFIDENCE_THRESHOLD,
        weather_service: WeatherService | None = None,
    ) -> None:
        self.explainer = explainer_agent
        self.threshold = threshold
        self.weather_service = weather_service or WeatherService()
        self.graph = self._build_graph()

    def _classify(self, state: CropDiseaseState) -> CropDiseaseState:
        # Image is resized to 256x256 and fed to the CNN model inside `model_api.predict`.
        predicted_disease, confidence = model_api.predict(state["image_path"])
        return {
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "explanation": None,
            "weather": None,
            "regional": None,
            "message": None,
        }

    @staticmethod
    def _extract_crop(predicted_disease: str) -> str:
        """Return the crop part of a 'Crop - Disease' label."""
        return predicted_disease.split(" - ", 1)[0]

    def _regional(self, state: CropDiseaseState) -> CropDiseaseState:
        """Assess crop survival/nativity/yield for the farmer's location."""
        regional = None
        message = state.get("message")
        location = state.get("location")
        predicted = state.get("predicted_disease")
        if location and predicted:
            crop = self._extract_crop(predicted)
            try:
                regional = self.explainer.regional_analysis(crop, location)
            except Exception as exc:  # noqa: BLE001 - never break the crop flow
                note = f"Regional assessment unavailable: {exc}"
                message = (message + "\n" + note).strip() if message else note
        return {"regional": regional, "message": message}

    def _weather(self, state: CropDiseaseState) -> CropDiseaseState:
        location = state.get("location")
        weather = None
        message = None
        if location:
            try:
                weather = self.weather_service.fetch(location)
                message = f"Weather fetched for {location}."
            except Exception as exc:  # noqa: BLE001 - never break the crop flow
                message = f"Weather unavailable: {exc}"
        return {"weather": weather, "message": message}

    def _explain(self, state: CropDiseaseState) -> CropDiseaseState:
        explanation = self.explainer.explain(
            disease_name=state["predicted_disease"],
            confidence=state["confidence"],
            weather=state.get("weather"),
        )
        return {"explanation": explanation}

    def _route(self, state: CropDiseaseState) -> str:
        confidence = state.get("confidence") or 0.0
        if confidence > self.threshold:
            return "explain"
        return "end"

    def _build_graph(self):
        graph = StateGraph(CropDiseaseState)

        graph.add_node("classify", self._classify)
        graph.add_node("weather", self._weather)
        graph.add_node("regional", self._regional)
        graph.add_node("explain", self._explain)

        graph.add_edge(START, "classify")
        graph.add_edge("classify", "weather")
        graph.add_edge("weather", "regional")
        graph.add_conditional_edges(
            "regional",
            self._route,
            {"explain": "explain", "end": END},
        )
        graph.add_edge("explain", END)

        return graph.compile()

    def run(
        self,
        image_path: str,
        location: Optional[str] = None,
    ) -> CropDiseaseState:
        """Run the full pipeline on an image and return the final state."""
        result = self.graph.invoke(
            {
                "image_path": image_path,
                "location": location,
                "predicted_disease": None,
                "confidence": None,
                "explanation": None,
                "weather": None,
                "regional": None,
                "message": None,
            }
        )
        return result