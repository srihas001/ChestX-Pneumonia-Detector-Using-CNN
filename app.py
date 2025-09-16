import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os

st.set_page_config(page_title="Chest X-ray Pneumonia Detector", layout="centered")

st.title("🩺 Chest X-ray Pneumonia Detector")
st.markdown("Upload a chest X-ray image, and the model will predict whether it shows **Pneumonia** or is **Normal**.")

MODEL_PATH = "model5.keras"
GDRIVE_URL = "https://drive.google.com/uc?id=1uiOX1hELu3gxIB0VBX4uc-lfoLBbjSu1"

if not os.path.exists(MODEL_PATH):
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((150, 150))
    x = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    prediction = model.predict(x)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded X-ray", use_container_width=True)
    with col2:
        st.markdown("### Prediction Result")
        st.markdown(f"**Label:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}")
        if label == "PNEUMONIA":
            st.markdown('<p style="color:red;font-size:20px">⚠️ Consult a Doctor!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green;font-size:20px">✅ Normal</p>', unsafe_allow_html=True)

st.markdown("---")
