import os
import shutil
import random
from pathlib import Path

# ==========================
# PATH DATASET ASLI
# ==========================
SOURCE_DIR = "Dataset sampah"

# OUTPUT
OUTPUT_DIR = "dataset_yolo"

# SPLIT
TRAIN_RATIO = 0.8

random.seed(42)

# ==========================
# KELAS UTAMA
# ==========================
CLASSES = [
    "Hazardous",
    "Non-Recyclable",
    "Organic",
    "Recyclable"
]

# ==========================
# BUAT FOLDER OUTPUT
# ==========================
for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, cls),
            exist_ok=True
        )

# ==========================
# PROSES DATA
# ==========================
for cls in CLASSES:

    class_path = os.path.join(SOURCE_DIR, cls)

    all_images = []

    # ambil semua gambar dari subfolder
    for root, dirs, files in os.walk(class_path):

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):

                full_path = os.path.join(root, file)
                all_images.append(full_path)

    print(f"{cls}: {len(all_images)} gambar")

    random.shuffle(all_images)

    split_idx = int(len(all_images) * TRAIN_RATIO)

    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # copy train
    for img in train_images:

        filename = Path(img).name

        dst = os.path.join(
            OUTPUT_DIR,
            "train",
            cls,
            filename
        )

        shutil.copy2(img, dst)

    # copy val
    for img in val_images:

        filename = Path(img).name

        dst = os.path.join(
            OUTPUT_DIR,
            "val",
            cls,
            filename
        )

        shutil.copy2(img, dst)

# ==========================
# DATA.YAML
# ==========================
yaml_content = """
path: dataset_yolo

train: train
val: val

names:
  0: Hazardous
  1: Non-Recyclable
  2: Organic
  3: Recyclable
"""

with open(
    os.path.join(OUTPUT_DIR, "data.yaml"),
    "w",
    encoding="utf-8"
) as f:
    f.write(yaml_content)

print("\nDataset berhasil dibuat!")
print(f"Lokasi: {OUTPUT_DIR}")