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

### Segmentation

Download example CamVid dataset to `data/segmentation/CamVid`:

```bash
uv run python3 scripts/download_example_data_segmentation.py
```

Run segmentation training:

```bash
dnn-train-segmentation
```

Run experiments:

```bash
dnn-train-segmentation --multirun model.name=resnet18,mobilenet_v2 train.epochs=5
```

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


## Loss functions

The loss functions (yes, plural) can be combined as you like. We use a wrapper which utilizes SMP's loss function module, which enables convenient loss functions initializations. The loss functions initialization is straight forward. Visit site [SMP losses](https://smp.readthedocs.io/en/latest/losses.html) and choose functions that you want your model to train on. Suppose we choose JaccardLoss and DiceLoss, and we want Jaccard to influence training more than Dice. We can add these two together and assign weights.

In the SMP losses docs we can see Jaccard and Dice are initialized with such params:

```python
class segmentation_models_pytorch.losses.DiceLoss(mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07, class_weights=None)

class segmentation_models_pytorch.losses.JaccardLoss(mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07, class_weights=None)
```

Our API enables you to declare those losses in the train/*.yaml configuration file like such:

```yaml
loss_function:
  - name: JaccardLoss
    weight: 0.8
    params:
      mode: multiclass
      ignore_index: 255
      eps: 1e-3
  - name: DiceLoss
    weight: 0.2
    params:
      mode: multiclass
      ignore_index: 255
      smooth: 1e-6
```

`params` will act as given SMP loss module class kwargs.

## Augmentations

Augmentation pipeline works in similar way. We use [Albumentations](https://pypi.org/project/albumentations/) for data augmentations. We define them in `augmentations/*.yaml` config files. 

Please refer to the albumentation class names and parameters in order to correctly define augmentation pipeline.

E.g. albumentations offer such [augmentation](https://albumentations.ai/explore/transform/Affine/docs/)

Which means we can use it in such way:

```yaml
- name: Affine
  params:
    p: 0.5
    translate_percent: [0.08, 0.08]
    scale: [0.85, 1.15]
    rotate: [-15, 15]
    border_mode: 4
    interpolation: 1
    fill: 0
```

`OneOf` augmentations can be composed like such:

```yaml
- name: OneOf
  params:
    p: 0.5
    transforms:
      - name: MotionBlur
        params:
          blur_limit: [3, 7]
      - name: MedianBlur
        params:
          blur_limit: [3, 7]
      - name: GaussianBlur
        params:
          blur_limit: [3, 7]
```

## Models

Model config offers a few useful options and settings:

* `freeze_backbone` - we freeze the backbone and leave the head as `requires_grad=True`, so only head has it's weights modified. It makes training way faster
* `pretrained` - use available pretrained weights, for better training
* `preprocessing` - a separate step from augmentation pipeline. Makes sure the images are in correct format to feed into the model, but you can put there anything you want until it's compatible with albumentations API

## Training 

The training config should be pretty easy and self explaining, except optimizer. We use Pytorch API as a wrapper to build the optimizers. So this code:

```yaml
optimizer:
  name: AdamW
  params:
    lr: 0.001
    weight_decay: 0.0001
```

translates roughly to: 

```python
from torch.optim import AdamW

optimizer = AdamW(lr=0.001, weight_decay=0.0001)
```

The training can always resume from checkpoint. Make sure to set the `resume_from_checkpoint` param to a string path to the checkpoint file. 


## Modules

Currently we support *classification* and *segmentation* module.

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

### Segmentaion

The `classification` module allows users to train vision models on custom datasets on a semantic segmentation task, that is per-pixel segmentaion.

User is able to compose various models and architectures, all listed on [Segmentation Models](https://smp.readthedocs.io/en/latest/) website.

Segmentation module requires user to have data in a strict structure, which consists of RGB images and masks:

```text
data/segmentation/<dataset>/
  train/images/*   train/masks/*
  val/images/*     val/masks/*
```

Images are pretty straightforward, but masks must be stored in RGB format as well, where each color represents one class. The color-class mapping must be included in the `configs/segmentation/data/*.yaml` file, otherwise the app won't know the class mapping and will raise an error.

Correct training produces useful charts. These include:

- Metric curves (loss and iou)
- Per class mIOU 
- Data distribution (pixel percentage for each class)
- Example images with and without augs
- Best model's predictions and annotations 


## Project layout

```text
configs/          Hydra YAML for each task
src/classification/
src/segmentation/
src/common/       Shared pipelines, tracking, export, experiment registry
src/core/         Base trainer, dataset, metrics
tests/unit/       Fast unit tests
tests/integration/ End-to-end training on synthetic data
```
