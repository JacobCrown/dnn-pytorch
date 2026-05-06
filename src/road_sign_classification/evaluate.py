from pathlib import Path

import numpy as np
from PIL import Image

from src.road_sign_classification.predictor import ClassificationPredictor

data_path = Path("data/classification/polish-road-signs/train")
checkpoint_path = Path(
    "output/classification/synthetic-road-signs/resnet34_128x128_100epochs_128bs/checkpoints/best.pth"
)

predictor = ClassificationPredictor(checkpoint_path)

total = 0
correct = 0

for image_path in data_path.glob("**/*.jpg"):
    image_class = image_path.parent.name
    image = np.array(Image.open(image_path))
    prediction = predictor.predict_class(image)
    total += 1
    if prediction == image_class:
        correct += 1
    else:
        pass
        # print(f"Prediction {prediction} does not match the image class {image_class}")

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.4f}")
