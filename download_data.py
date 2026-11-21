
from datasets import load_dataset
from pathlib import Path
import shutil

names = load_dataset(
    "ethz/food101", split="train"
).features["label"].names

for folder in ["data/images/train", "data/images/test"]:
    shutil.rmtree(folder, ignore_errors=True)

def prepare(split, limit, output):
    ds = load_dataset("ethz/food101", split=split)
    counts = {i: 0 for i in range(10)}

    for item in ds:
        label = item["label"]

        if label >= 10 or counts[label] >= limit:
            continue

        name = names[label]
        folder = Path("data/images") / output / name
        folder.mkdir(parents=True, exist_ok=True)

        item["image"].convert("RGB").save(
            folder / f"{counts[label]}.jpg"
        )
        counts[label] += 1

        if all(v == limit for v in counts.values()):
            break

    print(output, counts)

prepare("train", 100, "train")
prepare("validation", 30, "test")

print("All 10 classes downloaded!")
