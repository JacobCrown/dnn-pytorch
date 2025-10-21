from tests.helpers import TEST_ASSETS_DIR
from src.core.dataset import BaseDataset

assets_dir = TEST_ASSETS_DIR / "core"

class TestDataset:
    """Test if images are properly loaded and shaped as expected."""
    def test_load_image_rgba(self):
        image_path = assets_dir / "image_rgba.png"
        image = BaseDataset._load_image(image_path)
        assert image.shape == (100, 100, 3)

    def test_load_image_rgb(self):
        image_path = assets_dir / "image_rgb.png"
        image = BaseDataset._load_image(image_path)
        assert image.shape == (100, 100, 3)
        
    def test_load_image_gray(self):
        image_path = assets_dir / "image_gray.png"
        image = BaseDataset._load_image(image_path)
        assert image.shape == (100, 100, 1)