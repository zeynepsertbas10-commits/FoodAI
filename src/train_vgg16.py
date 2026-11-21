
import os
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.ao.quantization import quantize_dynamic


# ==========================================
# 1. SETTINGS
# ==========================================

DATA_DIR = "data/images/train"
MODEL_DIR = "models"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.0001
IMAGE_SIZE = 128
TEST_SPLIT = 0.20
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ==========================================
# 2. DATA TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. LOAD DATASET
# ==========================================

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

train_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=test_transform
)

classes = train_dataset.classes
num_classes = len(classes)

print("Classes:", classes)
print("Number of classes:", num_classes)
print("Total images:", len(train_dataset))

indices = list(range(len(train_dataset)))
random.shuffle(indices)

split_index = int(
    len(indices) * (1 - TEST_SPLIT)
)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

train_data = Subset(
    train_dataset,
    train_indices
)

test_data = Subset(
    test_dataset,
    test_indices
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_data))
print("Testing images:", len(test_data))


# ==========================================
# 4. LOAD PRETRAINED VGG16
# ==========================================

weights = models.VGG16_Weights.DEFAULT

model = models.vgg16(weights=weights)

# Freeze all convolutional layers first
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Fine-tune the final convolutional block
for parameter in model.features[24:].parameters():
    parameter.requires_grad = True

# Replace final classifier layer
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    num_classes
)

model = model.to(DEVICE)


# ==========================================
# 5. LOSS AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)

best_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())


# ==========================================
# 6. TRAINING
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    average_loss = running_loss / len(train_loader)


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


# ==========================================
# 7. SAVE BEST VGG16 MODEL
# ==========================================

model.load_state_dict(best_model_weights)
model.eval()

torch.save(
    model.state_dict(),
    os.path.join(
        MODEL_DIR,
        "vgg16.pth"
    )
)

print("\nBest VGG16 Accuracy:", best_accuracy)

print("VGG16 model saved.")


# ==========================================
# 8. DYNAMIC INT8 QUANTIZATION
# ==========================================

model_cpu = copy.deepcopy(model).cpu()
model_cpu.eval()

quantized_model = quantize_dynamic(
    model_cpu,
    {nn.Linear},
    dtype=torch.qint8
)

torch.save(
    quantized_model.state_dict(),
    os.path.join(
        MODEL_DIR,
        "vgg16_quantized.pth"
    )
)

print("Quantized VGG16 model saved.")

print("\nTraining completed successfully.")
