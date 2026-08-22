import os

import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    """Generate a synthetic model at MODEL_PATH if not present (CI environment)."""
    from src.model.architecture import SimpleCNN

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "cats_dogs_cnn.pt")
    )
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        m = SimpleCNN()
        torch.save(m.state_dict(), model_path)


@pytest.fixture(scope="session")
def synthetic_image_tensor():
    return torch.randn(1, 3, 224, 224)


@pytest.fixture(scope="session")
def synthetic_jpeg_file(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fixtures")
    path = tmp / "test_image.jpg"
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    img.save(str(path), format="JPEG")
    return str(path)


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as client:
        yield client
