import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

MODEL_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 224
DEFAULT_CLASS_NAMES = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Powdery Mildew",
    "Leaf Rust",
    "Bacterial Leaf Spot",
    "Leaf Curl",
    "Downy Mildew",
]

_model = None
_last_confidence = None


def build_cnn_model(num_classes: int) -> tf.keras.Model:
    """CNN with 5 conv blocks + 3 dense layers, Adam optimizer, accuracy metric."""
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            # Conv block 1
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, 2),
            # Conv block 2
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, 2),
            # Conv block 3
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, 2),
            # Conv block 4
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, 2),
            # Conv block 5
            layers.Conv2D(256, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            # Dense block 1
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),
            # Dense block 2
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.3),
            # Dense block 3 (output)
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(model_path: str | os.PathLike = None) -> tf.keras.Model:
    """Load the trained model once and reuse it across calls."""
    global _model
    if _model is None:
        path = Path(model_path) if model_path else MODEL_DIR / "crop_disease_model.keras"
        if path.exists():
            _model = tf.keras.models.load_model(path)
        else:
            # Fall back to a freshly built architecture so the pipeline can run.
            _model = build_cnn_model(len(DEFAULT_CLASS_NAMES))
    return _model


def predict(
    image_path: str | os.PathLike,
    class_names: list[str] | None = None,
) -> tuple[str, float]:
    """Classify an image and return (predicted_label, confidence)."""
    global _last_confidence
    model = load_model()
    class_names = class_names or DEFAULT_CLASS_NAMES

    image = tf.keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    array = tf.keras.utils.img_to_array(image) / 255.0
    batch = tf.expand_dims(array, axis=0)

    probabilities = model.predict(batch, verbose=0)[0]
    index = int(tf.argmax(probabilities))
    _last_confidence = float(probabilities[index])

    return class_names[index], _last_confidence


def giveConfidence() -> float:
    """Return the confidence of the most recent prediction."""
    if _last_confidence is None:
        raise RuntimeError("No prediction has been made yet. Call predict() first.")
    return _last_confidence
