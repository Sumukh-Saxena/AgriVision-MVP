from model.predictor import CropDiseasePredictor

_predictor = CropDiseasePredictor()


class ModelAPI:
    """Thin wrapper around the Keras predictor used by the workflow graph."""

    def __init__(self, predictor: CropDiseasePredictor | None = None) -> None:
        self._predictor = predictor or _predictor
        self._last_confidence: float | None = None

    def predict(self, image_path: str) -> tuple[str, float]:
        disease, confidence = self._predictor.predict(image_path)
        self._last_confidence = confidence
        return disease, confidence

    def giveConfidence(self) -> float:
        return self._last_confidence or 0.0


model = ModelAPI()
