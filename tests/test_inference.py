import torch

from src.model.architecture import SimpleCNN


def test_model_output_range(synthetic_image_tensor):
    model = SimpleCNN()
    model.eval()
    with torch.no_grad():
        out = model(synthetic_image_tensor)
    assert 0.0 <= out.item() <= 1.0


def test_model_output_shape(synthetic_image_tensor):
    model = SimpleCNN()
    model.eval()
    with torch.no_grad():
        out = model(synthetic_image_tensor)
    assert out.shape == torch.Size([1, 1])


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_predict_endpoint(api_client, synthetic_jpeg_file):
    with open(synthetic_jpeg_file, "rb") as f:
        response = api_client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("cat", "dog")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["inference_time_ms"] > 0
