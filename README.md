#For Dataset contact me

# Image-Classification
This project implements an end‑to‑end image classification pipeline in TensorFlow/Keras, trained on a custom dataset stored in Google Drive. It supports both pre‑trained CNN backbones and custom convolutional models.
**Image Classification Pipeline Documentation**

---

## Table of Contents

1. Project Overview
2. Key Features
3. Directory Structure
4. Dependencies & Installation
5. Configuration & Hyperparameters
6. Data Loading & Preprocessing
7. Model Architectures

   * Pre-trained CNNs (EfficientNetB0, DenseNet121, MobileNet)
   * Custom CNN Architectures (5 variants)
   * ResNet50 & VGG16 Building Functions
8. Training Workflow

   * Strategy Setup (TPU/CPU/GPU)
   * Checkpointing & Callbacks
   * Batch Size, Epochs, Optimizer
9. Usage Instructions
10. Expected Outcomes & Sample Results
11. SEO Keywords

---

## 1. Project Overview

This project implements an end‑to‑end image classification pipeline in TensorFlow/Keras, trained on a custom dataset stored in Google Drive. It supports both pre‑trained CNN backbones and custom convolutional models.

## 2. Key Features

* **Google Drive Integration:** Automatic mounting and model persistence.
* **Flexible Data Loader:** Reads images and metadata from CSV plus an image folder.
* **Image Preprocessing:** Resize to 128×128, normalize pixel values, one‑hot label encoding.
* **TPU/CPU/GPU Strategy:** Auto‑detect and distribute training.
* **Multiple Architectures:** Pre‑trained (EfficientNetB0, DenseNet121, MobileNet) and custom CNNs (5 variants).
* **Checkpointing:** Saves best model by validation accuracy.
* **Modular Design:** Clear separation of data loading, model creation, and training.

## 3. Directory Structure

```
project_root/
├── datasets.csv           # Metadata: image filename, category, objects, distance
├── images/                # Folder containing .jpg/.png files
├── models1/               # Output directory for saved .keras models
└── pipeline.ipynb         # Primary notebook/script implementing pipeline
```

## 4. Dependencies & Installation

```bash
pip install numpy pandas opencv-python scikit-learn tensorflow matplotlib
# TK and Colab integration already available
```

**TensorFlow Version:** 2.x
**Python Version:** 3.7+

## 5. Configuration & Hyperparameters

* **Image Size:** 128×128
* **Batch Size:** 32 for pre‑trained, 6 for custom models
* **Epochs:** 100
* **Optimizer:** Adam (lr=0.001)
* **Validation Split:** 20%

Modify paths at the top of `pipeline.ipynb`:

```python
image_folder_path = '/content/drive/My Drive/instabot/ds/images'
csv_path = '/content/drive/My Drive/instabot/ds/datasets.csv'
model_save_path = '/content/drive/My Drive/models1/'
```

## 6. Data Loading & Preprocessing

1. **`load_images_and_metadata()`** reads `datasets.csv`, matches filenames in `images/`, reads and resizes each image via OpenCV, normalizes to \[0,1], and collects labels and extra metadata.
2. **Label Encoding:** Unique categories → integer mapping → one‑hot vectors.
3. **Train/Test Split:** 80% train, 20% test via `train_test_split`.

## 7. Model Architectures

### 7.1 Pre‑trained CNNs

* **Backbones:** EfficientNetB0, DenseNet121, MobileNet (ImageNet weights)
* **Top Layers:** GlobalAveragePooling → Dense(256, ReLU) → Dense(num\_classes, Softmax)
* **Compilation:** `categorical_crossentropy`, metrics=`['accuracy']`

### 7.2 Custom CNN Variants

* **`build_model_1`–`build_model_5`**: Sequential CNNs with increasing depth, varying filter sizes, dropout rates.

### 7.3 ResNet50 & VGG16 Builders

* **`create_resnet()`** & **`create_vgg16()`**: Utility functions to create from scratch (no pre‑trained weights) with a custom classification head.

## 8. Training Workflow

1. **Strategy Setup:** Attempts TPU allocation, falls back to CPU/GPU.
2. **Model Creation & Compile:** Wrapped in `strategy.scope()`.
3. **Checkpointing:** `ModelCheckpoint` monitors `val_accuracy`, saves best weights.
4. **Fit Loop:** 100 epochs, batch size as configured, validation on test set.

## 9. Usage Instructions

1. **Mount Drive** (in Colab):

   python
   from google.colab import drive
   

drive.mount('/content/drive')

```
```
2. **Adjust Paths** at script top.
3. **Run Cells** sequentially:
- Data loading
- Label encoding & split
- Strategy detection
- Model training loops

## 10. Expected Outcomes & Sample Results
- **Training Curves:** ~90%+ validation accuracy on balanced small datasets.
- **Model Files:** `.keras` files in `models1/` folder (e.g. `EfficientNetB0.keras`).
- **Metrics:** Accuracy, loss logs printed per epoch.

## 11. SEO Keywords

```

image classification TensorFlow
pretrained CNN transfer learning
EfficientNet DenseNet MobileNet
custom CNN architectures
OpenCV dataset loader
TPU training strategy
Colab Google Drive integration
ModelCheckpoint Keras
image preprocessing pipeline
Python computer vision tutorial
```
