# 😟 Stress Detection and Remediation using Haar Cascade and CNN

This project aims to **detect stress levels** in individuals using **facial expression recognition**. It utilizes **Haar Cascade Classifiers** for face detection and a **Convolutional Neural Network (CNN)** for emotion classification. Based on the detected emotion, the system suggests **stress remediation techniques** like breathing exercises, music, or motivational quotes.


## 🧠 Problem Statement

With rising mental health concerns, detecting early signs of stress can be crucial. This system offers a non-invasive, real-time approach to detect stress through facial expressions and suggests helpful remedial actions.


## 🔧 Technologies Used

- **Python**
- **OpenCV** – for face detection (Haar Cascade)
- **TensorFlow / Keras** – for CNN model
- **NumPy, Matplotlib** – for data handling and visualization
- **Flask / Tkinter** – (Optional) for UI
- **Jupyter Notebook / Python Scripts** – for training and testing


## 🚀 How It Works

1. **Face Detection**: Uses Haar Cascade to detect faces in real-time from webcam.
2. **Emotion Classification**: The CNN model classifies the emotion (e.g., happy, sad, angry, stressed).
3. **Stress Detection**: If a stress-indicating emotion is detected, the system triggers a remediation module.
4. **Remediation Module**: Suggests calming activities like:
   - Music
   - Breathing exercise guide
   - Motivational quotes


## 🧪 Model Training

- Dataset: Labeled emotion dataset (like FER2013 or custom images)
- Preprocessing: Grayscale conversion, resizing, normalization
- Model: 4-layer CNN with ReLU, MaxPooling, and Softmax
- Accuracy: ~85% on validation set (can vary)
