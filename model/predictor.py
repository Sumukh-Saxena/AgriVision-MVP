import json
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = MODEL_DIR / "crop_disease_model.keras"
DEFAULT_CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
IMG_SIZE = (256, 256)


class CropDiseasePredictor:
    """Loads the Keras crop-disease model and predicts disease from an image."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        class_names_path: str | Path | None = None,
        img_size: tuple[int, int] = IMG_SIZE,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.class_names_path = Path(class_names_path or DEFAULT_CLASS_NAMES_PATH)
        self.img_size = img_size

        self.model = tf.keras.models.load_model(str(self.model_path))

        with open(self.class_names_path, encoding="utf-8") as f:
            self.class_names = json.load(f)

    def preprocess(self, image_path: str | Path) -> tf.Tensor:
        img = tf.keras.utils.load_img(str(image_path), target_size=self.img_size)
        img_array = tf.keras.utils.img_to_array(img)
        return tf.expand_dims(img_array, axis=0)

    def predict(self, image_path: str | Path) -> tuple[str, float]:
        """Return (predicted_disease, confidence) for the image at image_path."""
        img_array = self.preprocess(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        return self.class_names[predicted_index], confidence

    def predict_proba(self, image_path: str | Path) -> list[float]:
        """Return the full softmax probability vector for the image."""
        img_array = self.preprocess(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        return predictions[0].tolist()
