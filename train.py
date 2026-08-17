"""
Train a crop disease classifier from an ImageFolder-style dataset.

Expected structure:
data/
  train/
    Healthy/
    Tomato___Early_blight/
    Tomato___Late_blight/
    ...
  validation/
    Healthy/
    Tomato___Early_blight/
    ...

Run:
    python train.py
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
    raise FileNotFoundError(
        "Dataset folders not found. Create data/train and data/validation "
        "with one subfolder per disease class."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save(MODEL_PATH)

with open(os.path.join(MODEL_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

print(f"Model saved to: {MODEL_PATH}")
