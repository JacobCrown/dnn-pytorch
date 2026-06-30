# dnn-pytorch

Hydra-based PyTorch toolkit for image **classification** and semantic
**segmentation**. Shared training utilities live under `src/common`; task-specific
pipelines mirror each other in `src/classification` and `src/segmentation`.

## Features

- Classification (timm backbones) and segmentation (segmentation-models-pytorch)
- Hydra config composition for data, model, augmentations, and training
- Local checkpoints, logs, visualizations, and `run.json` experiment records
- Local JSON tracking (`tracking/events.jsonl`) with optional W&B config stub
- Experiment compare CLI (`dnn-compare`) and static HTML studio (`dnn-studio`)
- ONNX export and local JSON inference CLIs for both tasks
- Classification Optuna search and checkpoint evaluation

See [docs/TRAINING.md](docs/TRAINING.md) for full training workflows.

## Install

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

Verify before changing training code:

```bash
uv run pytest tests/unit -q
uv run ruff check .
```

End-to-end pipeline tests (synthetic data, ~few seconds):

```bash
uv run pytest tests/integration -m integration -q
```

## Quick start

Minimal classification smoke test (1 epoch, tiny model):

```bash
uv run python -m src.classification.main train=minimal model=minimal data.max_samples=20
```

Minimal segmentation smoke test:

```bash
uv run python -m src.segmentation.main train=minimal model=minimal data.max_samples=10
```

Default classification training:

```bash
uv run dnn-train-classification
```

Default segmentation training:

```bash
uv run dnn-train-segmentation
```

Compare local runs:

```bash
uv run dnn-compare output/
uv run dnn-studio output/
```

## Data layouts

**Classification** — folder per class:

```text
data/classification/<dataset>/
  train/<class>/*
  val/<class>/*
  test/<class>/*
```

**Segmentation** — paired images and RGB masks:

```text
data/segmentation/<dataset>/
  train/images/*   train/masks/*
  val/images/*     val/masks/*
```

## Run outputs

```text
output/classification/<dataset>/<model>_<HxW>_<epochs>epochs_<batch>bs/
output/segmentation/<dataset>/<arch>_<encoder>_<HxW>_<epochs>epochs_<batch>bs/
```

Each run includes checkpoints, `logs/main.log`, `run.json`, and (by default)
local tracking events.

## Console scripts

| Command | Purpose |
|---------|---------|
| `dnn-train-classification` | Train classifier |
| `dnn-train-segmentation` | Train segmenter |
| `dnn-search-classification` | Optuna hyperparameter search |
| `dnn-train-best-classification` | Retrain best Optuna trial |
| `dnn-compare` | Rank runs from `run.json` files |
| `dnn-studio` | HTML experiment leaderboard |
| `dnn-export-classification` / `dnn-export-segmentation` | ONNX export |
| `dnn-serve-classification` / `dnn-serve-segmentation` | Local JSON inference |

## Extension points

- Augmentations: `src/common/augmentations.py` + YAML
- Losses, optimizers, schedulers: `src/common`
- New tasks: follow `src/classification` / `src/segmentation` package shape

## Project layout

```text
configs/          Hydra YAML for each task
src/classification/
src/segmentation/
src/common/       Shared pipelines, tracking, export, experiment registry
src/core/         Base trainer, dataset, metrics
tests/unit/       Fast unit tests
tests/integration/ End-to-end training on synthetic data
docs/             Training and workflow documentation
```
