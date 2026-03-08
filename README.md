# 🌿 Plant Disease Classification using Deep Learning

## 📌 Project Overview
Plant diseases significantly affect agricultural productivity. Early detection can help farmers take preventive measures and reduce crop loss.

This project builds a **deep learning based image classification system** to automatically detect plant diseases from leaf images. Two models were implemented and compared:

- Custom Convolutional Neural Network (CNN)
- Transfer Learning using MobileNetV2

Additionally, **Grad-CAM visualization** is used to interpret the model predictions by highlighting important regions in the leaf images.

---

# 📂 Dataset

Dataset used: **PlantVillage Dataset**

Total Images: **20,638**

Total Classes: **15**

Classes include diseases from **Pepper, Potato, and Tomato plants**.

### Example Classes
- Pepper Bell Bacterial Spot
- Pepper Bell Healthy
- Potato Early Blight
- Potato Late Blight
- Potato Healthy
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Tomato Healthy

### Dataset Split

| Split | Size |
|------|------|
| Train | 14,446 |
| Validation | 3,095 |
| Test | 3,097 |

Split Ratio:

- **70% Training**
- **15% Validation**
- **15% Testing**

---

# ⚙️ Data Preprocessing

### Training Transformations
- Resize → 224 × 224
- Random Horizontal Flip
- Random Rotation (20°)
- Convert to Tensor
- Normalize using ImageNet mean & std

### Validation/Test Transformations
- Resize → 224 × 224
- Convert to Tensor
- Normalize using ImageNet mean & std

---

# 🧠 Models Implemented

## 1️⃣ Custom CNN Model

Architecture:

Input → **224 × 224 × 3**

### Convolution Layers
- Conv2D (3 → 32)
- Conv2D (32 → 64)
- Conv2D (64 → 128)

### Pooling
- MaxPooling (2 × 2)

### Fully Connected Layers
- FC1: 128 × 28 × 28 → 256
- Dropout: 0.5
- FC2: 256 → 15 classes

### Activation
ReLU

### Loss Function
CrossEntropyLoss

### Optimizer
Adam (learning rate = 0.001)

---

## 2️⃣ Transfer Learning: MobileNetV2

A pretrained **MobileNetV2 (ImageNet weights)** was used.

The feature extractor layers were **frozen** and only the classifier layer was trained.

### Modified Classifier

```python
nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
```

### Optimizer
Adam (learning rate = 1e-3)

---

# 📊 Training Results

## Custom CNN

| Metric | Value |
|------|------|
Train Accuracy | 85.8%
Validation Accuracy | 89.9%
Test Accuracy | **90.83%**

---

## MobileNetV2 (Transfer Learning)

| Metric | Value |
|------|------|
Train Accuracy | 89.4%
Validation Accuracy | 94.3%
Test Accuracy | **93.18%**

---

# 📈 Model Comparison

| Model | Test Accuracy |
|------|------|
Custom CNN | 90.83% |
MobileNetV2 | **93.18%** |

Transfer learning significantly improved the model performance due to pretrained ImageNet features.

---

# 📉 Evaluation Metrics

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Example Result:

```
Accuracy: 0.93
Macro Avg F1-score: 0.92
Weighted Avg F1-score: 0.93
```

---

# 🔍 Model Explainability (Grad-CAM)

Grad-CAM was implemented to visualize which regions of the leaf image influenced the model’s prediction.

### Steps

1. Identify the last convolutional layer of MobileNetV2
2. Compute gradients of the predicted class
3. Generate activation maps
4. Overlay heatmap on the original image

This helps verify that the model focuses on **disease-affected regions of the leaf**.

---

# 📊 Visualizations

The project includes:

- Training vs Validation Loss Curve
- Training vs Validation Accuracy Curve
- Confusion Matrix
- Grad-CAM Heatmap Visualization

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV

---

# 📁 Project Structure

```
Plant-Disease-Classification
│
├── dataset/
│
├── models/
│   ├── plant_disease_model.pth
│   ├── mobilenet_plant_disease_model.pth
│
├── notebooks/
│   └── training.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── gradcam.png
│
└── README.md
```

---

# 🚀 Future Improvements

Possible improvements:

- Train with **EfficientNet**
- Try **Vision Transformers (ViT)**
- Apply **advanced augmentation (MixUp, CutMix)**
- Handle **class imbalance**
- Deploy as **Streamlit web application**
- Build **mobile app for farmers**

---

# 🎯 Conclusion

This project demonstrates how **deep learning can be applied to agriculture** for automated plant disease detection.

Key takeaways:

- Transfer learning improves model performance
- MobileNetV2 achieved **93% accuracy**
- Grad-CAM helps make the model **interpretable**

Such systems can help farmers **detect plant diseases early and reduce crop loss**.
