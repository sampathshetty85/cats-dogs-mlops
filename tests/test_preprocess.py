import torch
from PIL import Image

from src.data.preprocess import EVAL_TRANSFORMS, TRAIN_TRANSFORMS


def _make_rgb_image(size=(256, 256)):
    return Image.new("RGB", size, color=(100, 150, 200))


def test_resize_output_shape():
    img = _make_rgb_image((300, 400))
    tensor = EVAL_TRANSFORMS(img)
    assert tensor.shape == torch.Size([3, 224, 224])


def test_normalize_range():
    img = _make_rgb_image()
    tensor = EVAL_TRANSFORMS(img)
    assert tensor.min().item() >= -3.0
    assert tensor.max().item() <= 3.0


def test_augmentation_preserves_shape():
    img = _make_rgb_image()
    tensor = TRAIN_TRANSFORMS(img)
    assert tensor.shape == torch.Size([3, 224, 224])
