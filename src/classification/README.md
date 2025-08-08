# Classification Module

A comprehensive PyTorch-based classification framework that provides everything you need to train and evaluate image classification models.

## Features

- **Dataset Management**: Easy-to-use dataset class for loading images organized by class folders
- **Data Augmentation**: Comprehensive augmentation pipeline with Mixup and CutMix support
- **Model Architectures**: Pre-built models including ResNet, EfficientNet, Vision Transformer, and custom CNN
- **Training Framework**: Complete training loop with validation, early stopping, and checkpointing
- **Configuration Management**: Type-safe configuration system using dataclasses
- **Utilities**: Helper functions for metrics, visualization, and experiment management

## Quick Start

### 1. Prepare Your Data

Organize your images in the following structure:

```
data/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
└── ...
```

### 2. Basic Usage

```python
from src.classification import (
    ClassificationDataset, get_train_transforms, get_val_transforms,
    create_model, ClassificationTrainer, get_default_config
)
from torch.utils.data import DataLoader
import torch.optim as optim

# Configuration
config = get_default_config()
config.model.name = "resnet18"
config.model.num_classes = 10
config.training.num_epochs = 50

# Create transforms
train_transforms = get_train_transforms(
    image_size=(224, 224),
    use_augmentation=True
)
val_transforms = get_val_transforms(image_size=(224, 224))

# Create datasets
train_dataset = ClassificationDataset(
    data_root="data/train",
    transforms=train_transforms
)
val_dataset = ClassificationDataset(
    data_root="data/val",
    transforms=val_transforms
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
model = create_model(
    model_name="resnet18",
    num_classes=10,
    pretrained=True
)

# Create optimizer and trainer
optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = ClassificationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer
)

# Train
history = trainer.train(num_epochs=50)
```

### 3. Advanced Usage with Configuration

```python
from src.classification import (
    ClassificationConfig, ClassificationDataset, create_model,
    ClassificationTrainer, create_data_loaders, split_dataset
)

# Load configuration
config = ClassificationConfig()
config.model.name = "efficientnet_b0"
config.data.batch_size = 64
config.training.learning_rate = 0.0001
config.training.num_epochs = 100

# Create full dataset and split
full_dataset = ClassificationDataset(
    data_root="data",
    transforms=get_train_transforms()
)

train_dataset, val_dataset, test_dataset = split_dataset(
    full_dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=config.data.batch_size
)

# Create model and trainer
model = create_model(
    model_name=config.model.name,
    num_classes=len(full_dataset.get_class_names()),
    pretrained=config.model.pretrained
)

trainer = ClassificationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda"
)

# Train and evaluate
history = trainer.train(num_epochs=config.training.num_epochs)
test_metrics = trainer.evaluate(test_loader)
```

## Components

### Dataset (`dataset.py`)

The `ClassificationDataset` class automatically discovers classes from folder names and loads images with their corresponding labels.

```python
dataset = ClassificationDataset(
    data_root="path/to/data",
    transforms=transforms,
    extensions=['.jpg', '.png', '.jpeg']
)

# Get class information
class_names = dataset.get_class_names()
class_to_idx = dataset.get_class_to_idx()
```

### Augmentations (`augmentations.py`)

Provides standard augmentation transforms and advanced techniques like Mixup and CutMix.

```python
# Standard transforms
train_transforms = get_train_transforms(
    image_size=(224, 224),
    use_augmentation=True
)

# Advanced augmentation
mixup_transform = MixupTransform(alpha=0.2)
cutmix_transform = CutMixTransform(alpha=1.0)
```

### Models (`models.py`)

Pre-built model architectures with easy customization.

```python
# Available models
models = [
    "resnet18", "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "vit_b_16", "vit_b_32", "vit_l_16",
    "simplecnn"
]

# Create model
model = create_model(
    model_name="resnet50",
    num_classes=1000,
    pretrained=True,
    dropout=0.2
)

# Get model information
info = get_model_info(model)
print(f"Parameters: {info['total_parameters']:,}")
```

### Trainer (`trainer.py`)

Complete training framework with built-in validation, checkpointing, and logging.

```python
trainer = ClassificationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device="cuda",
    save_dir="checkpoints",
    log_dir="logs"
)

# Train with early stopping
history = trainer.train(
    num_epochs=100,
    early_stopping_patience=10
)

# Evaluate
metrics = trainer.evaluate(test_loader)
```

### Configuration (`config.py`)

Type-safe configuration management using dataclasses.

```python
from src.classification import get_resnet_config

# Use predefined configurations
config = get_resnet_config()
config.training.num_epochs = 50
config.data.batch_size = 64

# Save and load configurations
config.save("config.json")
loaded_config = ClassificationConfig.load("config.json")
```

### Utilities (`utils.py`)

Helper functions for common tasks.

```python
from src.classification import (
    set_seed, calculate_metrics, plot_confusion_matrix,
    create_experiment_dir, get_device
)

# Set random seed
set_seed(42)

# Calculate metrics
metrics = calculate_metrics(y_true, y_pred, class_names)

# Plot confusion matrix
plot_confusion_matrix(
    metrics['confusion_matrix'],
    class_names,
    save_path="confusion_matrix.png"
)

# Create experiment directory
exp_dir = create_experiment_dir("experiments", "my_experiment")
```

## Example Script

Run the example script to see the complete workflow:

```bash
python src/classification/example.py
```

Make sure to organize your data in the required directory structure first.

## Supported Models

- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **EfficientNet**: efficientnet_b0 through efficientnet_b7
- **Vision Transformer**: vit_b_16, vit_b_32, vit_l_16, vit_l_32
- **Custom CNN**: SimpleCNN for lightweight models

## Data Augmentation

The framework supports various augmentation techniques:

- **Standard**: Random crop, flip, rotation, color jittering
- **Mixup**: Interpolates between pairs of images and labels
- **CutMix**: Cuts and pastes patches between images
- **Random Erasing**: Randomly erases rectangular regions

## Training Features

- **Automatic validation**: Built-in validation loop
- **Early stopping**: Stop training when validation performance plateaus
- **Checkpointing**: Save best model and periodic checkpoints
- **TensorBoard logging**: Automatic logging of metrics
- **Mixed precision**: Optional automatic mixed precision training
- **Gradient clipping**: Prevent gradient explosion

## Requirements

- PyTorch >= 2.7.0
- torchvision >= 0.22.0
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm
- tensorboard

## Tips

1. **Data Organization**: Ensure your data is properly organized in class-named folders
2. **Image Formats**: Supported formats include JPG, PNG, BMP, TIFF
3. **Batch Size**: Start with smaller batch sizes and increase based on your GPU memory
4. **Learning Rate**: Use the predefined configurations for different model architectures
5. **Augmentation**: Enable augmentation for training, disable for validation/testing
6. **Early Stopping**: Use early stopping to prevent overfitting
7. **Checkpointing**: Regularly save checkpoints to resume training if needed
