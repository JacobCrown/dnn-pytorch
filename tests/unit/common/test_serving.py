import json

from src.common.serving import (
    classification_response,
    segmentation_response,
    to_json,
)


def test_classification_response_contains_label_and_task() -> None:
    response = classification_response(label="stop", image_path="image.jpg")

    assert response == {
        "task": "classification",
        "image_path": "image.jpg",
        "label": "stop",
    }


def test_segmentation_response_contains_mask_shape() -> None:
    response = segmentation_response(mask_shape=(1, 1, 64, 64), image_path="image.png")

    assert response["task"] == "segmentation"
    assert response["mask_shape"] == [1, 1, 64, 64]


def test_to_json_outputs_sorted_json() -> None:
    payload = to_json({"b": 1, "a": 2})

    assert json.loads(payload) == {"a": 2, "b": 1}
    assert payload.endswith("\n")
