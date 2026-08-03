import json
from pathlib import Path

from omegaconf import ListConfig, OmegaConf

from src.common.experiment_registry import write_run_record
from src.common.serialization import to_plain_python
from src.common.training_contracts import TrainingArtifacts


def test_to_plain_python_converts_list_config() -> None:
    value = ListConfig([1, 2, {"mean": [0.1, 0.2, 0.3]}])

    plain = to_plain_python(value)

    assert plain == [1, 2, {"mean": [0.1, 0.2, 0.3]}]
    json.dumps(plain)


def test_to_plain_python_converts_nested_dict_config() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "preprocessing": [
                    {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406]}}
                ]
            }
        }
    )

    plain = to_plain_python(cfg)

    assert plain["model"]["preprocessing"][0]["params"]["mean"] == [
        0.485,
        0.456,
        0.406,
    ]
    json.dumps(plain)


def test_write_run_record_accepts_hydra_like_config(tmp_path: Path) -> None:
    config = {
        "augmentations": ListConfig([{"name": "HorizontalFlip", "params": {"p": 0.5}}]),
        "model": {
            "preprocessing": ListConfig(
                [{"name": "Normalize", "params": {"mean": [0.5, 0.5, 0.5]}}]
            )
        },
    }

    record_path = write_run_record(
        task_name="classification",
        artifacts=TrainingArtifacts.from_completed_run(tmp_path),
        metrics={"best_val_accuracy": 42.0},
        config=config,
        used_validation=True,
    )

    payload = json.loads(record_path.read_text())
    assert payload["config"]["augmentations"][0]["name"] == "HorizontalFlip"
    assert payload["config"]["model"]["preprocessing"][0]["params"]["mean"] == [
        0.5,
        0.5,
        0.5,
    ]
