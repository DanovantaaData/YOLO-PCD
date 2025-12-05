# 🍽️ Klasifikasi Makanan Padang

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Sistem klasifikasi otomatis untuk 9 jenis makanan khas Padang menggunakan teknik Pengolahan Citra Digital (PCD) dan Machine Learning**

</div>

---

## 📋 Deskripsi Project

Project ini merupakan implementasi **end-to-end pipeline Computer Vision** untuk klasifikasi makanan Padang. Sistem menggabungkan teknik-teknik klasik pengolahan citra dengan algoritma machine learning untuk mengidentifikasi 9 kategori makanan dengan akurasi tinggi.

### 🎯 **9 Kategori Makanan:**
1. **Ayam Goreng**
2. **Ayam Pop**
3. **Daging Rendang**
4. **Dendeng Batokok**
5. **Gulai Ikan**
6. **Gulai Tambusu**
7. **Gulai Tunjang**
8. **Telur Balado**
9. **Telur Dadar**

---

## 🌟 Fitur Utama

### **1. Preprocessing & Enhancement**
- ✅ **Image Resizing**: Normalisasi ukuran gambar ke 128×128
- ✅ **Gaussian Blur**: Noise reduction dengan kernel 5×5
- ✅ **Normalisasi**: Scaling pixel values ke range [0, 1]
- ✅ **CLAHE**: Contrast Limited Adaptive Histogram Equalization untuk enhancement
- ✅ **Color Space Conversion**: RGB, HSV, LAB untuk analisis multi-dimensional

### **2. Operasi Citra Klasik**
| Operasi | Teknik | Tujuan |
|---------|--------|--------|
| **Segmentasi** | Otsu Thresholding | Memisahkan objek makanan dari background |
| **Edge Detection** | Sobel, Canny, Laplacian | Mendeteksi tepi dan kontur |
| **Morfologi** | Erosion, Dilation, Opening, Closing | Cleanup noise & menutup gap |
| **Histogram Analysis** | RGB & HSV Channel Analysis | Memahami distribusi warna |

### **3. Feature Extraction (151 Features Total)**
#### **A. Color Features (105 features)**
- **Color Histogram**: 96 features (32 bins × 3 channels BGR)
- **Color Moments**: 9 features (Mean, Std Dev, Skewness × 3 channels)

#### **B. Texture Features (46 features)**
- **GLCM (Gray Level Co-occurrence Matrix)**: 20 features
  - Properties: Contrast, Dissimilarity, Homogeneity, Energy, Correlation
  - 4 orientasi: 0°, 45°, 90°, 135°
- **LBP (Local Binary Pattern)**: 26 features
  - Uniform pattern dengan radius=1, points=8

### **4. Machine Learning Classification**
- **Algoritma**: K-Nearest Neighbors (KNN)
- **Konfigurasi Optimal**:
  - `n_neighbors = 8` (hasil hyperparameter tuning)
  - `weights = 'distance'` (bobot berbasis jarak)
  - `metric = 'euclidean'`
- **Preprocessing ML**: StandardScaler untuk normalisasi fitur
- **Data Split**: 80% Training, 10% Validation, 10% Testing

---

## 🛠️ Teknologi & Library

```python
# Core Libraries
import cv2                    # OpenCV untuk operasi citra
import numpy as np            # Numerical computing
import matplotlib.pyplot as plt  # Visualisasi

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Feature Extraction
from skimage.feature import graycomatrix, graycoprops  # GLCM
from skimage.feature import local_binary_pattern       # LBP
```

---

## 📦 Instalasi

### **Prerequisites**
- Python 3.8 atau lebih tinggi
- Google Colab (opsional, untuk notebook version)
- GPU (opsional, untuk training lebih cepat)

### **Setup Environment**

```bash
# Clone repository
git clone https://github.com/DanovantaaData/YOLO-PCD.git
cd YOLO-PCD

# Install dependencies
pip install -r requirements.txt
```

### **requirements.txt**
```txt
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
matplotlib>=3.4.0
seaborn>=0.11.0
pillow>=8.3.0
```

---

## 📁 Struktur Dataset

```
dataset_padang_food/
├── ayam_goreng/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── ayam_pop/
│   └── ...
├── daging_rendang/
│   └── ...
├── dendeng_batokok/
│   └── ...
├── gulai_ikan/
│   └── ...
├── gulai_tambusu/
│   └── ...
├── gulai_tunjang/
│   └── ...
├── telur_balado/
│   └── ...
└── telur_dadar/
    └── ...
```

**Format Gambar**: `.jpg`, `.jpeg`, `.png`  
**Rekomendasi**: Minimal 50-100 gambar per kategori untuk hasil optimal

---

## 🚀 Cara Menggunakan

### **1. Training Model (Google Colab)**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set path dataset
DATASET_PATH = '/content/drive/MyDrive/dataset_padang_food'

# Jalankan notebook sel per sel
# atau gunakan:
# !jupyter nbconvert --execute klasifikasi_makanan_padang.ipynb
```

### **2. Training Model (Local)**

```python
# Import library
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset
DATASET_PATH = 'dataset_padang_food'
# ... (ikuti kode di notebook)

# Training
knn_model.fit(X_train_scaled, y_train)

# Save model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
```

### **3. Inferensi / Prediksi**

```python
import cv2
import pickle
import numpy as np

# Load model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load & preprocess image baru
img = cv2.imread('test_image.jpg')
img_resized = cv2.resize(img, (128, 128))
img_denoised = cv2.GaussianBlur(img_resized, (5, 5), 0)
img_normalized = img_denoised / 255.0

# Extract features
features = extract_all_features(img_normalized)
features_scaled = scaler.transform([features])

# Prediksi
prediction = model.predict(features_scaled)
class_name = label_encoder.inverse_transform(prediction)[0]

print(f"Prediksi: {class_name}")
```

---

## 📊 Hasil Evaluasi

### **Performance Metrics**

| Metric | Nilai |
|--------|-------|
| **Akurasi Overall** | **~62.03%** |
| **Precision (Rata-rata)** | **~66%** |
| **Recall (Rata-rata)** | **~61%** |
| **F1-Score (Rata-rata)** | **~62%** |

### **Confusion Matrix**

```
                    Predicted
                AG   AP   DR   DB   GI   GT   GJ   TB   TD
Actual  AG       4    0    1    2    1    0    0    0    0
        AP       0    5    0    0    0    1    0    0    2
        DR       0    0    6    2    0    0    0    0    1
        DB       2    0    1    6    0    0    0    0    0
        GI       0    0    0    0    6    2    0    0    1
        GT       0    0    0    1    1    5    1    0    1
        GJ       0    0    0    0    1    1    7    0    0
        TB       1    0    1    1    0    0    1    3    0
        TD       1    2    0    0    0    1    0    0    7
```

**Legenda**: AG=Ayam Goreng, AP=Ayam Pop, DR=Daging Rendang, DB=Dendeng Batokok, GI=Gulai Ikan, GT=Gulai Tambusu, GJ=Gulai Tunjang, TB=Telur Balado, TD=Telur Dadar

### **Detail Performa Per Kelas**

| Kategori | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Ayam Goreng** | 50% | 50% | 50% | 8 |
| **Ayam Pop** | 71% | 62% | 67% | 8 |
| **Daging Rendang** | 67% | 67% | 67% | 9 |
| **Dendeng Batokok** | 50% | 67% | 57% | 9 |
| **Gulai Ikan** | 67% | 67% | 67% | 9 |
| **Gulai Tambusu** | 50% | 56% | 53% | 9 |
| **Gulai Tunjang** | **78%** | **78%** | **78%** | 9 |
| **Telur Balado** | **100%** | 43% | 60% | 7 |
| **Telur Dadar** | 58% | 64% | 61% | 11 |

**Kelas Terbaik**: Gulai Tunjang (78% F1-score) & Telur Balado (100% precision)  
**Kelas Tersulit**: Ayam Goreng & Gulai Tambusu (50-53% F1-score)  
**Catatan**: Gulai variants memiliki similarity tinggi sehingga lebih challenging untuk diklasifikasi.

---

## 📊 Visualisasi Pipeline

### **1. Preprocessing Flow**

```
Original Image (Arbitrary Size)
         ↓
[Resize] → 128×128
         ↓
[Gaussian Blur] → Kernel 5×5
         ↓
[Normalization] → [0, 1]
         ↓
[CLAHE Enhancement] → Improved Contrast
```

### **2. Feature Extraction Flow**

```
Preprocessed Image
         ↓
    ┌────┴────┐
    ↓         ↓
  COLOR    TEXTURE
    ↓         ↓
┌───┴───┐  ┌──┴──┐
│ Hist  │  │GLCM │
│(96)   │  │(20) │
│       │  │     │
│Moment │  │ LBP │
│(9)    │  │(26) │
└───┬───┘  └──┬──┘
    └────┬────┘
         ↓
   151 Features
         ↓
  StandardScaler
         ↓
    KNN Classifier
         ↓
   Classification
```