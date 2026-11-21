
import os
import time

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.ao.quantization import quantize_dynamic


# ==========================================
# 1. SETTINGS
# ==========================================

DEVICE = torch.device("cpu")
TEST_DIR = "data/images/test"

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

torch.set_num_threads(1)


# ==========================================
# 2. IMAGE TRANSFORMS
# ==========================================

cnn_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

vgg_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. TEST DATASET
# ==========================================

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_DIR}"
    )

cnn_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=cnn_transform
)

vgg_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=vgg_transform
)

if cnn_dataset.classes != vgg_dataset.classes:
    raise ValueError("CNN and VGG16 class names do not match.")

classes = cnn_dataset.classes
num_classes = len(classes)

cnn_loader = DataLoader(
    cnn_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

vgg_loader = DataLoader(
    vgg_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Classes:", classes)
print("Number of classes:", num_classes)
print("Test images:", len(cnn_dataset))


# ==========================================
# 4. CNN ARCHITECTURE
# ==========================================

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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==========================================
# 5. LOAD MODELS
# ==========================================

def load_models():

    models_list = []

    # CNN
    cnn = CNN()

    cnn.load_state_dict(
        torch.load(
            "models/cnn.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    cnn.eval()

    models_list.append(
        ("CNN", cnn, cnn_loader, "models/cnn.pth")
    )

    # Quantized CNN
    cnn_q = CNN()

    cnn_q = quantize_dynamic(
        cnn_q,
        {nn.Linear},
        dtype=torch.qint8
    )

    cnn_q.load_state_dict(
        torch.load(
            "models/cnn_quantized.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    cnn_q.eval()

    models_list.append(
        ("CNN Quantized", cnn_q, cnn_loader,
         "models/cnn_quantized.pth")
    )

    # VGG16
    vgg = models.vgg16(weights=None)

    vgg.classifier[6] = nn.Linear(
        vgg.classifier[6].in_features,
        num_classes
    )

    vgg.load_state_dict(
        torch.load(
            "models/vgg16.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    vgg.eval()

    models_list.append(
        ("VGG16", vgg, vgg_loader, "models/vgg16.pth")
    )

    # Quantized VGG16
    vgg_q = models.vgg16(weights=None)

    vgg_q.classifier[6] = nn.Linear(
        vgg_q.classifier[6].in_features,
        num_classes
    )

    vgg_q = quantize_dynamic(
        vgg_q,
        {nn.Linear},
        dtype=torch.qint8
    )

    vgg_q.load_state_dict(
        torch.load(
            "models/vgg16_quantized.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    vgg_q.eval()

    models_list.append(
        ("VGG16 Quantized", vgg_q, vgg_loader,
         "models/vgg16_quantized.pth")
    )

    return models_list


# ==========================================
# 6. MODEL EVALUATION
# ==========================================

def evaluate(model, loader):

    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            # Top-1 Accuracy
            predictions = outputs.argmax(dim=1)

            correct_top1 += (
                predictions == labels
            ).sum().item()

            # Top-5 Accuracy
            top5_predictions = outputs.topk(
                min(5, outputs.shape[1]),
                dim=1
            ).indices

            correct_top5 += (
                top5_predictions == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            total += labels.size(0)

    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total

    return top1_accuracy, top5_accuracy


# ==========================================
# 7. LATENCY BENCHMARK
# ==========================================

def benchmark_latency(model, loader):

    model.eval()

    # Use one test image
    images, _ = next(iter(loader))
    image = images[0:1].to(DEVICE)

    # Warm-up
    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):
            model(image)

    # 100 inference measurements
    times = []

    with torch.inference_mode():

        for _ in range(BENCHMARK_RUNS):

            start = time.perf_counter()

            model(image)

            end = time.perf_counter()

            times.append((end - start) * 1000)

    average_latency = sum(times) / len(times)

    return average_latency


# ==========================================
# 8. MODEL SIZE
# ==========================================

def get_model_size(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    size_bytes = os.path.getsize(path)

    return size_bytes / (1024 * 1024)


# ==========================================
# 9. MAIN COMPARISON
# ==========================================

def main():

    print("\n===================================")
    print("       FOODAI MODEL COMPARISON")
    print("===================================")

    print("Device:", DEVICE)
    print("Test images:", len(cnn_dataset))
    print("Warm-up runs:", WARMUP_RUNS)
    print("Benchmark runs:", BENCHMARK_RUNS)

    results = []

    for name, model, loader, model_path in load_models():

        print(f"\nEvaluating: {name}")

        top1, top5 = evaluate(
            model,
            loader
        )

        latency = benchmark_latency(
            model,
            loader
        )

        size = get_model_size(model_path)

        results.append({
            "name": name,
            "top1": top1,
            "top5": top5,
            "latency": latency,
            "size": size
        })

        print(f"Top-1 Accuracy: {top1:.2f}%")
        print(f"Top-5 Accuracy: {top5:.2f}%")
        print(f"Model Size: {size:.2f} MB")
        print(f"Average Latency: {latency:.3f} ms")

    # ======================================
    # 10. FINAL RESULTS TABLE
    # ======================================

    print("\n\n===================================")
    print("          FINAL RESULTS")
    print("===================================")

    print(
        f"{'Model':<22}"
        f"{'Top-1 %':<12}"
        f"{'Top-5 %':<12}"
        f"{'Size MB':<12}"
        f"{'Latency ms':<12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['name']:<22}"
            f"{result['top1']:<12.2f}"
            f"{result['top5']:<12.2f}"
            f"{result['size']:<12.2f}"
            f"{result['latency']:<12.3f}"
        )

    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()
