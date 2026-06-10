# Segmentation

Hydra-driven semantic segmentation using [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch).

## Entry points

| Command | Module | Purpose |
|---------|--------|---------|
| `dnn-train-segmentation` | `src.segmentation.main` | Train with Hydra configs |
| `dnn-test-segmentation` | `src.segmentation.test` | Evaluate a checkpoint on the test split |
| `dnn-export-segmentation` | `src.segmentation.export` | Export checkpoint to ONNX |
| `dnn-serve-segmentation` | `src.segmentation.serve` | Local JSON inference (+ optional mask PNG) |

Programmatic training without Hydra:

```python
from src.segmentation.config import SegmentationConfig
from src.segmentation.training_pipeline import run_segmentation_training

config = SegmentationConfig.from_yaml("path/to/config.yaml")
run_segmentation_training(config)
```

## Data layout

```text
data/segmentation/<dataset>/
  train/images/*
  train/masks/*
  val/images/*
  val/masks/*
  test/images/*    # optional
  test/masks/*
```

Masks are RGB PNGs. Class colors and optional merges are defined in `configs/segmentation/data/default.yaml` (`class_config`, `class_mapping`).

## Key modules

| File | Role |
|------|------|
| `config.py` | Dataclass configs (`SegmentationConfig`, `SegmentationTestConfig`) |
| `dataset.py` | Paired image/mask dataset with RGB→class lookup |
| `mask_utils.py` | Mask preparation and RGB conversion helpers |
| `model_manager.py` | SMP model construction and checkpoint loading |
| `training_pipeline.py` | Datasets → trainer → checkpoints → `run.json` |
| `trainer.py` | Epoch loop, mIoU tracking, evaluation helpers |
| `predictor.py` | Standalone inference from checkpoint preprocessing |
| `visualizer.py` | Pre/post-training overlays and metric plots |

## Config groups

Hydra configs under `configs/segmentation/`:

- `main.yaml` — training composer
- `test.yaml` — checkpoint evaluation
- `train/default.yaml`, `model/default.yaml`, `data/default.yaml`
- `train=minimal`, `model=minimal` — fast smoke tests

## Outputs

Each run writes under `output/segmentation/<dataset>/<arch>_<encoder>_<HxW>_<epochs>epochs_<batch>bs/`:

- `checkpoints/best.pth`, `last.pth`
- `logs/main.log`
- `run.json`, `tracking/events.jsonl`

Checkpoints store `class_names`, `class_idx_to_rgb`, and `class_idx_to_name` for inference.

See [docs/TRAINING.md](../../docs/TRAINING.md) for full workflows.
