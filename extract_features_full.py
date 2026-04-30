import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
img_size = 224
batch_size = 64
data_dir = r"Organize/train"
model = load_model("crop_disease_model.h5")
print("Model loaded successfully!")

feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-2].output
)

print("Feature extractor ready!")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2   # MUST match your training split
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# =============================
# Extract Features
# =============================
print("Extracting TRAIN features...")
train_features = feature_extractor.predict(train_generator)
train_labels = train_generator.classes

print("Extracting VALIDATION features...")
val_features = feature_extractor.predict(val_generator)
val_labels = val_generator.classes

# =============================
# Save
# =============================
np.save("train_features.npy", train_features)
np.save("train_labels.npy", train_labels)

np.save("val_features.npy", val_features)
np.save("val_labels.npy", val_labels)

print("Feature extraction completed successfully!")