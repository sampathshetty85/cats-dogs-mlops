import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def __repr__(self):
        lines = [
            "SimpleCNN",
            "  Input : (B, 3, 224, 224)",
            "  Conv1 : 3→32, k=3 p=1 → BN → ReLU → MaxPool2d(2)",
            "  Conv2 : 32→64, k=3 p=1 → BN → ReLU → MaxPool2d(2)",
            "  Conv3 : 64→128, k=3 p=1 → BN → ReLU → AdaptiveAvgPool(7,7)",
            "  FC1   : 128×7×7=6272 → 512 → ReLU → Dropout(0.5)",
            "  FC2   : 512 → 1 → Sigmoid",
            "  Output: (B, 1)  threshold=0.5  (≥0.5 → dog, <0.5 → cat)",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    model = SimpleCNN()
    print(model)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"
    assert 0.0 <= out.item() <= 1.0, f"Output {out.item()} not in [0,1]"
    print(f"\nForward pass OK — output: {out.item():.4f}")
