# 🩺 Chest X-ray Pneumonia Detector

A deep learning application that uses Convolutional Neural Networks (CNN) to detect pneumonia from chest X-ray images. The model is deployed as a user-friendly Streamlit web application.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

## 🔍 Overview

This project implements a binary classification system to distinguish between normal chest X-rays and those showing signs of pneumonia. The CNN model is trained on a dataset of chest X-ray images and achieves high accuracy in pneumonia detection.

## ✨ Features

- **Real-time Prediction**: Upload chest X-ray images and get instant predictions
- **User-friendly Interface**: Clean and intuitive Streamlit web interface
- **High Accuracy**: CNN model with 84% test accuracy
- **Confidence Scores**: Provides prediction confidence levels
- **Visual Feedback**: Clear indicators for normal vs pneumonia cases
- **Medical Guidance**: Automatic recommendations to consult doctors for positive cases

## 🏗️ Model Architecture

The CNN model consists of:

```
- Input Layer: (150, 150, 3) - RGB images
- Conv2D Layer 1: 32 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pooling
- Conv2D Layer 2: 64 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pooling
- Conv2D Layer 3: 128 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pooling
- Flatten Layer
- Dense Layer: 128 neurons, ReLU activation
- Dropout: 0.5 rate
- Output Layer: 1 neuron, Sigmoid activation (Binary classification)
```

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 10
- Batch Size: 32
- Image Size: 150x150 pixels

## 📊 Dataset

The model is trained on the Chest X-ray Images (Pneumonia) dataset with the following distribution:

- **Training Set**: 5,216 images (2 classes)
- **Validation Set**: 16 images (2 classes)
- **Test Set**: 624 images (2 classes)

**Classes:**
- Normal: Healthy chest X-rays
- Pneumonia: X-rays showing pneumonia symptoms

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ChestX-Pneumonia-Detector-Using-CNN-2.git
   cd ChestX-Pneumonia-Detector-Using-CNN-2
   ```

2. **Install required packages:**
   ```bash
   pip install streamlit tensorflow pillow numpy matplotlib
   ```

   Or create a requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Web Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Upload a chest X-ray image** using the file uploader

4. **View the prediction results** including:
   - Predicted label (Normal/Pneumonia)
   - Confidence score
   - Medical recommendation

### Using the Model Directly

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("model5.keras")

# Load and preprocess image
img = Image.open("chest_xray.jpg").convert("RGB")
img_resized = img.resize((150, 150))
x = np.array(img_resized)
x = np.expand_dims(x, axis=0) / 255.0

# Make prediction
prediction = model.predict(x)[0][0]
label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}")
```

## 📈 Model Performance

- **Training Accuracy**: 93.98%
- **Validation Accuracy**: 75.00%
- **Test Accuracy**: 84.46%
- **Loss**: 0.51 (test set)

### Training History
The model shows good convergence with:
- Steady improvement in training accuracy
- Stable validation performance
- Low overfitting tendency

## 📁 Project Structure

```
ChestX-Pneumonia-Detector-Using-CNN-2/
│
├── app.py                 # Streamlit web application
├── model5.keras          # Trained CNN model
├── model.h5              # Alternative model format
├── Untitled.ipynb        # Jupyter notebook with model training
├── README.md             # Project documentation
│
└── chest_xray/           # Dataset directory (not included in repo)
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Web Framework**: Streamlit
- **Image Processing**: PIL (Pillow)
- **Data Manipulation**: NumPy
- **Visualization**: Matplotlib
- **Development**: Jupyter Notebook

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This application is for educational and research purposes only. It is NOT intended for:
- Clinical diagnosis
- Medical decision making
- Replacement of professional medical advice

**Always consult with qualified healthcare professionals for medical diagnosis and treatment decisions.**

The model's predictions should be considered as preliminary screening tools and not as definitive medical diagnoses. False positives and false negatives are possible, and the model's performance may vary with different types of chest X-ray images.
