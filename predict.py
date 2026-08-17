import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/crop_disease_model.keras"
CLASS_FILE = "models/class_names.txt"

if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/leaf.jpg")
    raise SystemExit(1)

if not os.path.exists(MODEL_PATH):
    print("Model not found. Run train.py first.")
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

image = Image.open(sys.argv[1]).convert("RGB").resize((224, 224))
arr = np.array(image, dtype=np.float32) / 255.0
arr = np.expand_dims(arr, axis=0)

pred = model.predict(arr, verbose=0)[0]
idx = int(np.argmax(pred))

print("Disease:", class_names[idx])
print("Confidence: {:.2f}%".format(float(pred[idx]) * 100))
