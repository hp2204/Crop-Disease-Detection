import os
import shutil
import random

# 🔹 Path to your extracted dataset (CHANGE THIS)
SOURCE_DIR = SOURCE_DIR = r"D:\plantvillage dataset\color"


# 🔹 Output folder
OUTPUT_DIR = r"D:\organize"

# Split ratio
TRAIN_RATIO = 0.8

# Create main folders
for split in ["train", "test"]:
    for category in ["Healthy", "Diseased"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)

# Loop through all class folders
for class_folder in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_folder)

    if not os.path.isdir(class_path):
        continue

    # Decide category
    if "healthy" in class_folder.lower():
        category = "Healthy"
    else:
        category = "Diseased"

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Copy files
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(OUTPUT_DIR, "train", category, img)
        shutil.copy2(src, dst)

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(OUTPUT_DIR, "test", category, img)
        shutil.copy2(src, dst)

print("Dataset organized successfully!")
