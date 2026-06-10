# dnn-pytorch

`dnn-pytorch` is a Hydra-based PyTorch toolkit for learning and iterating on
vision models. The active modules today are image classification and semantic
segmentation; both use shared training utilities under `src/common` and base
interfaces under `src/core`.

## What Works Today

- Classification training with timm backbones.
- Segmentation training with segmentation-models-pytorch backbones.
- Hydra config composition for data, model, training, and augmentations.
- Local checkpoints, logs, and visualizations under `output/`.
- Classification Optuna search and retraining from best trials.
- Classification checkpoint evaluation with `src.classification.test`.

The project is intentionally local-first. Cloud tracking, distributed training,
model export, experiment comparison, and serving are planned but not yet part of
the stable workflow.

## Install

This project is managed with `uv` and targets Python 3.13 or newer.

```bash
uv sync
```

Run the unit suite and linter before changing training code:

```bash
uv run pytest tests/unit -q
uv run ruff check .
```

## Classification

Train with the default classification config:

```bash
uv run dnn-train-classification
```

Override config groups or fields with Hydra syntax:

```bash
uv run dnn-train-classification model=default train=default data=default
uv run dnn-train-classification model.name=resnet18 model.height=224 model.width=224
```

Run Optuna search and retrain the best trial:

```bash
uv run dnn-search-classification
uv run dnn-train-best-classification
```

Evaluate a saved classification checkpoint:

```bash
uv run python -m src.classification.test \
  eval_checkpoint=/absolute/path/to/output/classification/.../checkpoints/best.pth
```

Classification expects a folder-per-class layout:

```text
data/classification/<dataset>/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

## Segmentation

Train with the default segmentation config:

```bash
uv run dnn-train-segmentation
```

Segmentation expects image and mask folders per split:

```text
data/segmentation/<dataset>/
  train/images/*.png
  train/masks/*.png
  val/images/*.png
  val/masks/*.png
  test/images/*.png
  test/masks/*.png
```

Class colors and optional class merging are configured in
`configs/segmentation/data/default.yaml`.

## Outputs

Training runs write artifacts to deterministic local directories:

```text
output/classification/<dataset>/<model>_<height>x<width>_<epochs>epochs_<batch>bs/
output/segmentation/<dataset>/<architecture>_<encoder>_<height>x<width>_<epochs>epochs_<batch>bs/
```

Each run may include:

- `checkpoints/last.pth`
- `checkpoints/best.pth`
- `checkpoints/predictor.pth` for classification inference
- `logs/run.log`
- `visualizations/*.png`

## Extension Points

- Add new augmentations through `src/common/augmentations.py` and YAML config.
- Add losses, optimizers, and schedulers through `src/common`.
- Add new model families by implementing a task model manager.
- Add new tasks by following the package shape in `src/classification` and
  `src/segmentation`.

## Known Gaps

- Segmentation orchestration is still concentrated in `src/segmentation/main.py`.
- There is no distributed training strategy yet.
- ONNX dependencies are installed, but no export CLI exists yet.
- W&B is available but normal training does not use a tracking abstraction yet.
- There is no local experiment comparison, model registry, or serving layer yet.
