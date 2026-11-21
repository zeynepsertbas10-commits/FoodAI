
import torch
import torch.nn as nn
from torchvision import models
from torch.ao.quantization import quantize_dynamic

# CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Load models
cnn = CNN()
cnn.load_state_dict(torch.load("models/cnn.pth", map_location="cpu", weights_only=True))

vgg = models.vgg16(weights=None)
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 10)
vgg.load_state_dict(torch.load("models/vgg16.pth", map_location="cpu", weights_only=True))

# Quantization
for name, model in [("cnn", cnn), ("vgg16", vgg)]:
    model.eval()

    quantized = quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    torch.save(
        quantized.state_dict(),
        f"models/{name}_quantized.pth"
    )

    print(name, "quantized successfully!")

print("Quantization completed!")
