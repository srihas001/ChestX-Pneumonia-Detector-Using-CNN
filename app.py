import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import os

st.set_page_config(page_title="Chest X-ray Pneumonia Detector", layout="centered")

st.title("🩺 Chest X-ray Pneumonia Detector")
st.markdown("Upload a chest X-ray image and choose which model you want to use.")

# -------------------------------
# MODEL INFO
# -------------------------------

MODEL_A_NAME = "model5.keras"
MODEL_A_URL = "https://drive.google.com/uc?id=1uiOX1hELu3gxIB0VBX4uc-lfoLBbjSu1"

MODEL_B_NAME = "pneumonia_transfer_model.h5"
MODEL_B_URL = "https://drive.google.com/uc?id=1Gd9M9AMidFwVYD_mnGDkbzRejff6C1dy"

# -------------------------------
# DOWNLOAD MODELS IF MISSING
# -------------------------------

if not os.path.exists(MODEL_A_NAME):
    st.info("Downloading Basic CNN model...")
    gdown.download(MODEL_A_URL, MODEL_A_NAME, quiet=False)

if not os.path.exists(MODEL_B_NAME):
    st.info("Downloading Transfer Learning model...")
    gdown.download(MODEL_B_URL, MODEL_B_NAME, quiet=False)

# -------------------------------
# MODEL SELECTOR
# -------------------------------

model_choice = st.selectbox(
    "Choose a model:",
    ["Basic CNN Model (model5.keras)", "Transfer Learning Model (pneumonia_transfer_model.h5)"]
)

selected_model_path = MODEL_A_NAME if "Basic" in model_choice else MODEL_B_NAME

# Load the selected model
model = load_model(selected_model_path)

# -------------------------------
# IMAGE UPLOAD
# -------------------------------

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

def preprocess(img, model):
    img = img.convert("RGB")
    
    # auto-detect required input size
    try:
        shape = model.input_shape
        size = (shape[1], shape[2])
    except:
        size = (224, 224)

    img_resized = img.resize(size)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, size

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    x, used_size = preprocess(img, model)

    prediction = float(model.predict(x)[0][0])
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption=f"Uploaded X-ray (Resized to {used_size})", use_container_width=True)

    with col2:
        st.markdown("### Prediction Result")
        st.markdown(f"**Model Used:** {model_choice}")
        st.markdown(f"**Label:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}")

        if label == "PNEUMONIA":
            st.warning("""
### ⚠️ Possible Pneumonia Detected
This is **not** a diagnosis. Recommended next steps:

- Get a **doctor evaluation** soon.
- Take a **professional chest X-ray report**.
- Monitor **fever, cough, and breathing difficulty**.
- Check **oxygen (SpO₂)** regularly.
  - If SpO₂ < **92%**, seek urgent care.
- Rest, hydrate, avoid physical exertion.
- **Do NOT** take antibiotics/steroids without a prescription.
""")
        else:
            st.success("This X-ray looks NORMAL based on the model.")
            st.info("""
If you still have symptoms (fever, cough, breathlessness),
please consult a doctor. Machine predictions are not medical confirmation.
""")

st.markdown("---")
st.caption("This application is for educational/research use only.")
