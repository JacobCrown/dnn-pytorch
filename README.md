# dnn-pytorch: High level API for training Vision Models

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)![Apache 2.0 license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

This project allows end users to:

* Create various vision models for classification and segmentation tasks
* Train models and experiment with configs
* Keep track of training metrics with MLFlow
* Build augmentation pipelines and preprocessing
* Merge classes based on config
* Run multiple experiments with a single line of code

## Install

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Quick start

### Classification

Download example CIFAR-10 dataset to `data/classification/cifar-10`:

```bash
uv run python3 scripts/download_example_data_classification.py
```

Run classification training:

```bash
dnn-train-classification
```

Look into output/classification for checkpoints, logs and visualizations.

Run experiments:

```bash
dnn-train-classification --multirun model.name=resnet18 train.epochs=10 train.batch_size=128,256
```

The code above will run 2 experimets varying in batch_size parameter.

## Configs

This project uses [hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/index.html#) for managing config. 

It is recommended to use custom config rather than simply modifying the default.yaml files.

E.g. we want to run training on different model than the default one. We can create a resnet18.yaml config file in configs/*/model/ :

```yaml
name: "resnet18"
in_channels: 3
pretrained: true
... # Rest is as in default.yaml file
```

In order to run training on our new config we can simply execute:

```bash
dnn-train-classification model=resnet18
```

We are able to modify all configurable modules, that is model, augmentations, train and dataset however we want and then run it. 

## Modules

Currently we support *classification* module.

Check down below for in detail explanation and how to run experiments with different configurations

### Classification

The `classification` module allows users to train vision models on custom datasets on a classification task.

The models user is able to train are all models from the [timm](https://timm.fast.ai/) library.

If you're not sure what are the available models, run:

```bash
uv run python3 -c "import timm; print(timm.list_models(\"*convnext*\"))"
```

Which will print all models containing 'convnext' inside model architecture name. 

For obtaining all models, run:

```bash
uv run python3 -c "import timm; print(timm.list_models())"
```

but the list might be long.

Classification module requires user to have data in a strict structure:

```text
data/classification/<dataset>/
  train/<class>/*
  val/<class>/*
  test/<class>/*
```

Look inside example CIFAR-10 folder structure. The class names will be derived automatically from the folder structure. The folder structure will be checked while composing the dataset and errors will be raised if composed incorrectly. 

Correct training produces useful charts. These include:

- Metric curves
- Confusion matrix
- Example images with and without augs
- Best model's predictions and annotations 

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
