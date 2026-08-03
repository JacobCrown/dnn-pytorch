# Classification

Hydra-driven image classification on top of [timm](https://github.com/huggingface/pytorch-image-models) backbones.

## Entry points

| Command | Module | Purpose |
|---------|--------|---------|
| `dnn-train-classification` | `src.classification.main` | Train with Hydra configs |
| `dnn-test-classification` | `src.classification.test` | Evaluate a checkpoint on the test split |
| `dnn-search-classification` | `src.classification.optuna_search` | Optuna hyperparameter search |
| `dnn-train-best-classification` | `src.classification.train_best` | Retrain the best Optuna trial |
| `dnn-export-classification` | `src.classification.export` | Export checkpoint to ONNX |
| `dnn-serve-classification` | `src.classification.serve` | Local JSON inference |

Programmatic training without Hydra:

```python
from src.classification.config import ClassificationConfig
from src.classification.training_pipeline import run_classification_training

config = ClassificationConfig.from_yaml("path/to/config.yaml")
run_classification_training(config)
```

## Data layout

```text
data/classification/<dataset>/
  train/<class_name>/*
  val/<class_name>/*
  test/<class_name>/*
```

`ClassificationDataset` discovers classes from folder names. Optional `class_mapping` merges source folders into one label (see `configs/classification/data/default.yaml`).

## Key modules

| File | Role |
|------|------|
| `config.py` | Dataclass configs (`ClassificationConfig`, `ClassificationTestConfig`) |
| `dataset.py` | Folder-per-class dataset |
| `model_manager.py` | timm model construction, freeze policy, checkpoint loading |
| `training_pipeline.py` | Datasets → trainer → checkpoints → `run.json` |
| `trainer.py` | Epoch loop, AMP, EMA, predictor checkpoint export |
| `predictor.py` | Standalone inference from `predictor.pth` or `best.pth` |
| `visualizer.py` | Pre/post-training plots under `visualizations/` |

## Config groups

Hydra configs live under `configs/classification/`:

- `main.yaml` — training composer
- `test.yaml` — checkpoint evaluation
- `train/default.yaml`, `model/default.yaml`, `data/default.yaml`
- `train=minimal`, `model=minimal` — fast smoke tests

## Outputs

Each run writes under `output/classification/<dataset>/<model>_<HxW>_<epochs>epochs_<batch>bs/`:

- `checkpoints/best.pth`, `last.pth`, `predictor.pth`
- `logs/main.log`
- `run.json` — experiment record for `dnn-compare` / `dnn-studio`
- `tracking/events.jsonl` — per-epoch metrics

See [docs/TRAINING.md](../../docs/TRAINING.md) for full workflows.
