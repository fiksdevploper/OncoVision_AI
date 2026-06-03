# 🔬 OncoVision AI — Breast Cancer Detection with SVM

<p align="center">
  <img src="https://img.shields.io/badge/Model-SVM-blue?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-98.25%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>
</p>

<p align="center">
  Sistem deteksi kanker payudara berbasis Machine Learning menggunakan algoritma <strong>Support Vector Machine (SVM)</strong>,
  dibangun di atas dataset klinis Wisconsin Breast Cancer dan disajikan melalui REST API dengan FastAPI.
</p>

---

## 📌 Deskripsi Project

**OncoVision AI** adalah proyek machine learning yang bertujuan untuk mendeteksi kanker payudara secara otomatis berdasarkan fitur-fitur klinis dari hasil biopsi sel tumor. Model ini mengklasifikasikan tumor sebagai **ganas (malignant)** atau **jinak (benign)** dengan tingkat akurasi **98.25%**.

Model dilatih menggunakan dataset `Breast Cancer Wisconsin (Diagnostic)` dari `sklearn.datasets` yang memuat 30 fitur numerik hasil komputasi citra digital biopsi jarum halus (FNA) dari massa payudara.

---

## 🇮🇩 Latar Belakang & Urgensi

Kanker payudara merupakan ancaman kesehatan serius di Indonesia. Berikut fakta berdasarkan data publik resmi:

| Indikator | Data |
|---|---|
| Kasus baru kanker payudara (Globocan 2020) | **68.858 kasus** (16,6% dari total kanker nasional) |
| Estimasi kasus baru per tahun di Indonesia | **~65.000 kasus/tahun** |
| Angka kematian akibat kanker payudara | **>22.000 jiwa/tahun** |
| Persentase terdeteksi di stadium lanjut | **~70% kasus** |
| Perempuan usia 30–50 yg jalani deteksi dini (2022–2024) | **10,84 juta (25,55% dari target)** |
| Beban biaya kanker (BPJS 2020) | **Rp 3,5 triliun** (terbesar ke-2 setelah jantung) |

> 📊 *Sumber: Kementerian Kesehatan RI — Profil Kesehatan Indonesia 2024, Globocan 2020*

Hampir **70% pasien kanker terdeteksi pada stadium lanjut**, yang berdampak signifikan terhadap prognosis dan biaya pengobatan. Deteksi dini berbasis AI dapat menjadi alat bantu skrining yang cepat, murah, dan scalable untuk menjawab gap layanan kesehatan ini.

---

## 🧪 Dataset

```python
from sklearn.datasets import load_breast_cancer

kangker = load_breast_cancer()
```

| Info | Detail |
|---|---|
| Total sampel | 569 data |
| Fitur | 30 fitur numerik (radius, texture, perimeter, area, dll.) |
| Target | `0` = Malignant (ganas), `1` = Benign (jinak) |
| Sumber | UCI Machine Learning Repository via scikit-learn |

---

## 📊 Hasil Evaluasi Model

### Confusion Matrix

```
---Hasil Klasifikasi Algoritma SVM---
[[45  2]
 [ 0 67]]
```

|  | Predicted: Ganas (0) | Predicted: Jinak (1) |
|---|---|---|
| **Actual: Ganas (0)** | ✅ 45 (True Negative) | ❌ 2 (False Positive) |
| **Actual: Jinak (1)** | ❌ 0 (False Negative) | ✅ 67 (True Positive) |

> ⚠️ **False Negative = 0** — Model tidak melewatkan satu pun kasus kanker ganas yang sebenarnya. Ini krusial dalam konteks medis.

### Classification Report

```
              precision    recall  f1-score   support

           0       1.00      0.96      0.98        47
           1       0.97      1.00      0.99        67

    accuracy                           0.98       114
   macro avg       0.99      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114

Akurasi: 98.25%
```

### Ringkasan Performa

| Metrik | Kelas 0 (Malignant) | Kelas 1 (Benign) | Overall |
|---|---|---|---|
| **Precision** | 1.00 | 0.97 | 0.98 |
| **Recall** | 0.96 | 1.00 | 0.98 |
| **F1-Score** | 0.98 | 0.99 | 0.98 |
| **Accuracy** | — | — | **98.25%** |

---

<!-- Ganti placeholder di bawah ini dengan screenshot asli -->
<!-- ![App Screenshot](assets/screenshot_main.png) -->
<!-- ![API Response](assets/screenshot_api.png) -->

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|---|---|
| **ML / Data Science** | `scikit-learn`, `pandas`, `numpy`, `joblib` |
| **Visualisasi** | `matplotlib`, `seaborn` |
| **API** | `FastAPI` |
| **Notebook** | `Google Colab` |
| **Model Serialization** | `joblib` |

---

## 📁 Struktur Repository

```
oncovision-ai/
├── Cancer_Detection.ipynb     # Notebook utama (Google Colab)
├── model/
│   └── svm_model.pkl          # Model SVM yang sudah dilatih
├── app/
│   ├── main.py                # FastAPI endpoint
│   └── schemas.py             # Request/Response schema
├── assets/
│   └── screenshot_*.png       # Screenshot aplikasi
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/username/oncovision-ai.git
cd oncovision-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan API
```bash
uvicorn app.main:app --reload
```

### 4. Akses Dokumentasi API
Buka browser dan buka: `http://localhost:8000/docs`

### 5. Atau jalankan Notebook
Buka `Cancer_Detection.ipynb` di [Google Colab](https://colab.research.google.com/)

---

## 📡 API Endpoint

### `POST /predict`

**Request Body:**
```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, ...]
}
```

**Response:**
```json
{
  "prediction": 0,
  "label": "Malignant",
  "confidence": 0.98
}
```

---

## 📦 Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
fastapi
uvicorn
```

---

## 👤 Author

**Fikri** — ML Engineer Intern  
[![GitHub](https://img.shields.io/badge/GitHub-fiksdev-181717?style=flat&logo=github)](https://github.com/fiksdev)
[![Portfolio](https://img.shields.io/badge/Portfolio-fiksdev.pages.dev-blue?style=flat&logo=cloudflare)](https://fiksdev.pages.dev)

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for early cancer detection in Indonesia
</p>
