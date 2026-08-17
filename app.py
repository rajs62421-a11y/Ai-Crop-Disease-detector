import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Crop Disease Prediction", page_icon="🌱")
st.title("🌱 Crop Disease Prediction")
st.write("Upload a crop leaf image to predict its disease.")

MODEL_PATH = "models/crop_disease_model.keras"

# Demo fallback labels. Replace/update these after training.
CLASS_NAMES = [
    "Healthy",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Potato - Early Blight",
    "Potato - Late Blight"
]

@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        pass
    return None

model = load_model()
uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if model is None:
        st.warning(
            "No trained model was found. Add a trained model at "
            "`models/crop_disease_model.keras`, then run the app again."
        )
    else:
        img = image.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        predictions = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(predictions))
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
        confidence = float(predictions[idx]) * 100

        st.success(f"Prediction: **{label}**")
        st.metric("Confidence", f"{confidence:.2f}%")
