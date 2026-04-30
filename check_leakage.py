import os
import hashlib

def get_image_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

train_dir = "dataset/train"
val_dir = "dataset/validation"

train_hashes = set()

# Hash all train images
for root, _, files in os.walk(train_dir):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(root, file)
            train_hashes.add(get_image_hash(path))

duplicates = []

# Compare validation images
for root, _, files in os.walk(val_dir):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(root, file)
            img_hash = get_image_hash(path)
            if img_hash in train_hashes:
                duplicates.append(path)

print("Number of duplicate images:", len(duplicates))

if duplicates:
    print("Sample duplicate:", duplicates[0])
