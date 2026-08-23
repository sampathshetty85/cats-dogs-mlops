import os

import pytest
import torch
from PIL import Image

from src.config import IMAGE_SIZE, MODEL_PATH


@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    """Generate a synthetic model at MODEL_PATH if not present (CI environment)."""
    from src.model.architecture import SimpleCNN

    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        m = SimpleCNN()
        torch.save(m.state_dict(), MODEL_PATH)


@pytest.fixture(scope="session")
def model():
    from src.model.architecture import SimpleCNN
    m = SimpleCNN()
    m.eval()
    return m


@pytest.fixture(scope="session")
def synthetic_image_tensor():
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture(scope="session")
def synthetic_jpeg_file(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fixtures")
    path = tmp / "test_image.jpg"
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 64, 32))
    img.save(str(path), format="JPEG")
    return str(path)


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as client:
        yield client
