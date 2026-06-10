# Training Guide

This document describes how to train, evaluate, and iterate on models in
`dnn-pytorch` for both **classification** and **segmentation**.

## Quick smoke test

Use the minimal config groups when you want a fast end-to-end check (1 epoch,
small model, no AMP):

```bash
uv run python -m src.classification.main train=minimal model=minimal data.max_samples=20
uv run python -m src.segmentation.main train=minimal model=minimal data.max_samples=10
```

Integration tests under `tests/integration/` run the same pipelines on synthetic
tiny datasets without Hydra.

## Classification

### Data layout

```text
data/classification/<dataset>/
  train/<class_name>/*.{jpg,png,...}
  val/<class_name>/*.{jpg,png,...}
  test/<class_name>/*.{jpg,png,...}   # optional
```

Set `data.max_samples` to cap samples per split while debugging.

### Train

```bash
uv run python -m src.classification.main
uv run python -m src.classification.main model.name=resnet18 train.epochs=10
uv run python -m src.classification.main train=minimal model=minimal
```

Console entry point (same Hydra config):

```bash
uv run dnn-train-classification
```

### Evaluate a checkpoint

```bash
uv run python -m src.classification.test \
  eval_checkpoint=/absolute/path/to/checkpoints/best.pth
```

Evaluate a saved segmentation checkpoint:

```bash
uv run dnn-test-segmentation \
  eval_checkpoint=/absolute/path/to/checkpoints/best.pth
```

### Optuna search

```bash
uv run dnn-search-classification
uv run dnn-train-best-classification
```

## Segmentation

### Data layout

```text
data/segmentation/<dataset>/
  train/images/*
  train/masks/*
  val/images/*
  val/masks/*
  test/images/*    # optional
  test/masks/*
```

Class colors and optional merges live in `configs/segmentation/data/default.yaml`.

### Train

```bash
uv run python -m src.segmentation.main
uv run dnn-train-segmentation
```

Override fields the same way as classification (`train.epochs=5`, `data.max_samples=10`, …).

## Run artifacts

Each completed training run writes:

```text
output/<task>/<dataset>/<run_name>/
  checkpoints/best.pth
  checkpoints/last.pth
  checkpoints/predictor.pth     # classification only
  logs/main.log
  tracking/events.jsonl         # local JSON metrics/artifacts
  run.json                      # machine-readable run record
  visualizations/               # when enabled in config
```

`run.json` powers comparison and the local experiment studio.

## Experiment tooling

Compare runs:

```bash
uv run dnn-compare output/classification output/segmentation
```

Generate a static HTML leaderboard:

```bash
uv run dnn-studio output/
```

Export and serve (after training):

```bash
uv run dnn-export-classification output/.../checkpoints/best.pth model.onnx
uv run dnn-serve-classification output/.../checkpoints/predictor.pth image.jpg
```

Segmentation equivalents: `dnn-export-segmentation`, `dnn-serve-segmentation`.

## Tracking

Local JSON tracking is enabled by default (`train.tracking.local_enabled=true`).
Each epoch appends metrics to `tracking/events.jsonl`; a final summary is logged
after training completes.

Optional W&B is configured via `train.tracking.wandb_enabled=true` and
`wandb_project`. When W&B is enabled but not fully wired, the pipeline falls
back to local JSON and logs a warning.

## Testing

```bash
# Fast unit suite
uv run pytest tests/unit -q

# End-to-end training on synthetic data (slower)
uv run pytest tests/integration -m integration -q

# Everything
uv run pytest tests/unit tests/integration -q
uv run ruff check .
```

## Troubleshooting

### `ListConfig is not JSON serializable` after training

Fixed in `src/common/serialization.py`. Hydra/OmegaConf values in configs are
converted to plain Python before writing `run.json` or tracking events. If you
see this again, ensure you are on a version that includes `to_plain_python()`.

### Hydra `train=minimal` breaks output paths

Minimal train configs must still define `root_dir`, `checkpoints_dir`, and
related fields because Hydra replaces the entire `train` group. Use the provided
`configs/classification/train/minimal.yaml` as a template.

### Segmentation resumes from an old checkpoint unexpectedly

Ensure `train.resume_from_checkpoint` is `null` in your config unless you
intentionally continue training.
