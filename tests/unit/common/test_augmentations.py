import pytest
from pathlib import Path
import yaml
from typing import List

from src.common.config import AugmentationConfig
from src.common.augmentations import get_augmentations


@pytest.fixture
def augmentations() -> List[AugmentationConfig]:
    assets_path = Path(__file__).parent / "assets" / "augmentations.yaml"
    with open(assets_path, "r") as f:
        return [AugmentationConfig.from_dict(aug) for aug in yaml.safe_load(f)]


class TestAugmentations:
    def test_augmentations_are_correctly_initialized(self, augmentations):
        """Tests that augmentations are initialized with correct parameters from config."""
        augmentations_albumentations = get_augmentations(augmentations)
        for aug_alb, aug in zip(augmentations_albumentations.transforms, augmentations):
            assert aug_alb.__class__.__name__ == aug.name