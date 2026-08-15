import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.predictor import CropDiseasePredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict crop disease from a leaf/crop image."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the input crop/leaf image (JPG or PNG)",
    )
    args = parser.parse_args()

    predictor = CropDiseasePredictor()
    disease, confidence = predictor.predict(args.image)

    print(f"Predicted: {disease} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()
