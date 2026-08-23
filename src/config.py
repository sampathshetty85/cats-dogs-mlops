import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(_REPO_ROOT, "models", "cats_dogs_cnn.pt")

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Maps raw dataset directory names → output label names
CLASSES = {"Cat": "cat", "Dog": "dog"}
