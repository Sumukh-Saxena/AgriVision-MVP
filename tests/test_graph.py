"""Tests for the LangGraph workflow weather integration (worker/agents/graph.py).

The heavy CNN model is mocked so these tests are fast and offline.
"""

import unittest
from unittest import mock

try:
    from worker.agents.graph import CropDiseaseWorkflow, CropDiseaseState
    LANGGRAPH_AVAILABLE = True
except ModuleNotFoundError:  # langgraph not installed in this env
    LANGGRAPH_AVAILABLE = False
    CropDiseaseWorkflow = None
    CropDiseaseState = None


class FakeExplainer:
    def __init__(self):
        self.calls = []
        self.regional_calls = []

    def explain(self, disease_name, confidence, weather=None):
        self.calls.append({"disease_name": disease_name, "confidence": confidence, "weather": weather})
        return "explanation text"

    def regional_analysis(self, crop, location):
        self.regional_calls.append({"crop": crop, "location": location})
        return "regional report"


class FakeWeatherService:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {
            "city": "Pune", "country": "IN", "temperature_c": 28.5, "humidity": 62,
        }
        self.error = error

    def fetch(self, location):
        if self.error:
            raise self.error
        return self.payload


@mock.patch("worker.agents.graph.model_api")
class WorkflowWeatherTest(unittest.TestCase):
    def setUp(self):
        if not LANGGRAPH_AVAILABLE:
            self.skipTest("langgraph is not installed; skipping graph tests.")

    def _make(self, weather_service, explainer):
        return CropDiseaseWorkflow(
            explainer_agent=explainer,
            weather_service=weather_service,
            threshold=0.5,
        )

    def test_weather_node_fetches_and_routes_to_explain(self, mock_model):
        mock_model.predict.return_value = ("Tomato - Late Blight", 0.93)
        explainer = FakeExplainer()
        wf = self._make(FakeWeatherService(), explainer)
        state = wf.run("img.png", location="Pune")
        self.assertEqual(state["predicted_disease"], "Tomato - Late Blight")
        self.assertIsNotNone(state["weather"])
        self.assertEqual(state["weather"]["city"], "Pune")
        self.assertEqual(state["explanation"], "explanation text")
        self.assertEqual(explainer.calls[0]["weather"]["city"], "Pune")

    def test_regional_assessment_runs_for_location(self, mock_model):
        mock_model.predict.return_value = ("Tomato - Late Blight", 0.93)
        explainer = FakeExplainer()
        wf = self._make(FakeWeatherService(), explainer)
        state = wf.run("img.png", location="Pune")
        self.assertEqual(state["regional"], "regional report")
        self.assertEqual(explainer.regional_calls[0]["crop"], "Tomato")
        self.assertEqual(explainer.regional_calls[0]["location"], "Pune")

    def test_regional_skipped_without_location(self, mock_model):
        mock_model.predict.return_value = ("Tomato - Late Blight", 0.93)
        explainer = FakeExplainer()
        wf = self._make(FakeWeatherService(), explainer)
        state = wf.run("img.png")
        self.assertIsNone(state["regional"])
        self.assertEqual(explainer.regional_calls, [])

    def test_regional_failure_does_not_break_flow(self, mock_model):
        mock_model.predict.return_value = ("Tomato - Late Blight", 0.93)
        explainer = FakeExplainer()
        explainer.regional_analysis = lambda crop, location: (_ for _ in ()).throw(RuntimeError("LLM down"))
        wf = self._make(FakeWeatherService(), explainer)
        state = wf.run("img.png", location="Pune")
        self.assertIsNone(state["regional"])
        self.assertEqual(state["predicted_disease"], "Tomato - Late Blight")
        self.assertIn("Regional assessment unavailable", state["message"])

    def test_low_confidence_skips_explanation_but_keeps_weather(self, mock_model):
        mock_model.predict.return_value = ("Tomato - Healthy", 0.20)
        explainer = FakeExplainer()
        wf = self._make(FakeWeatherService(), explainer)
        state = wf.run("img.png", location="Pune")
        self.assertIsNone(state["explanation"])
        self.assertIsNotNone(state["weather"])
        self.assertIsNotNone(state["regional"])
        self.assertEqual(explainer.calls, [])

    def test_weather_failure_does_not_break_flow(self, mock_model):
        mock_model.predict.return_value = ("Apple - Healthy", 0.90)
        explainer = FakeExplainer()
        wf = self._make(FakeWeatherService(error=ValueError("bad key")), explainer)
        state = wf.run("img.png", location="Pune")
        self.assertEqual(state["predicted_disease"], "Apple - Healthy")
        self.assertIsNone(state["weather"])
        self.assertIn("Weather unavailable", state["message"])
        # explanation still runs without weather context
        self.assertEqual(state["explanation"], "explanation text")
        self.assertIsNone(explainer.calls[0]["weather"])

    def test_no_location_skips_weather_cleanly(self, mock_model):
        mock_model.predict.return_value = ("Grape - Healthy", 0.91)
        explainer = FakeExplainer()
        wf = self._make(FakeWeatherService(), explainer)
        state = wf.run("img.png")
        self.assertIsNone(state["weather"])
        self.assertIsNone(state["message"])
        self.assertEqual(state["explanation"], "explanation text")
        self.assertIsNone(explainer.calls[0]["weather"])


if __name__ == "__main__":
    unittest.main()