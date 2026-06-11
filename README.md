# 🔬 OncoVision AI — Clinical Decision Support System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>
</p>

<p align="center">
  <strong>Sistem Pendukung Keputusan Klinis Berbasis Multimodal AI</strong> — mengintegrasikan analisis citra medis
  dan data tabular klinis dalam satu dasbor terpadu untuk mendukung diagnosis dini kanker dan penyakit ginjal.
</p>

---

## 📌 Overview

**OncoVision AI** adalah platform klinis berbasis kecerdasan buatan yang dirancang untuk membantu tenaga medis dalam proses skrining dan deteksi dini penyakit secara cepat dan akurat. Sistem ini mengintegrasikan tiga modul diagnostik utama dalam satu antarmuka terpadu:

- **Deteksi Batu Ginjal** — menggunakan object detection berbasis YOLOv8 untuk melokalisasi dan menghitung batu ginjal pada citra medis (USG/CT Scan).
- **Klasifikasi Tumor Otak** — menggunakan model deep learning berbasis MobileNetV2 (Transfer Learning) untuk mengklasifikasikan jenis tumor pada citra MRI.
- **Analisis Risiko Kanker Payudara** — menggunakan klasifikasi data tabular klinis (CSV) untuk mengidentifikasi tingkat risiko keganasan tumor.

Sistem ini dibangun di atas **FastAPI** sebagai backend ringan dan performatif, dengan antarmuka frontend yang responsif dan visualisasi data interaktif menggunakan **Chart.js**.

---

## ✨ Key Features

- 🧠 **Multimodal AI Pipeline** — mendukung input berupa citra medis (JPEG/PNG) dan data tabular (CSV) dalam satu sistem terintegrasi.
- 🔍 **Object Detection (YOLOv8)** — mendeteksi lokasi, ukuran, dan jumlah batu ginjal secara real-time dari citra medis.
- 🩻 **Brain Tumor Classification** — klasifikasi MRI tumor otak menggunakan arsitektur MobileNetV2 dengan Transfer Learning.
- 📊 **Tabular Risk Analysis** — analisis data klinis CSV untuk estimasi risiko kanker payudara.
- ⚡ **Real-Time Inference** — inferensi model AI dengan latensi rendah melalui endpoint REST API FastAPI.
- 📈 **Interactive Dashboard** — visualisasi hasil diagnostik dengan Chart.js, disajikan dalam UI klinis yang bersih dan informatif.
- 📄 **Structured Reporting** — output laporan analisis yang dapat diinterpretasikan oleh tenaga medis.

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|---|---|
| **Bahasa Pemrograman** | Python 3.10+ |
| **Backend / API** | FastAPI, Uvicorn (ASGI Server) |
| **Object Detection** | YOLOv8 (`ultralytics`) |
| **Deep Learning** | TensorFlow / Keras, MobileNetV2 (Transfer Learning) |
| **Computer Vision** | OpenCV (`opencv-python`) |
| **Data Processing** | NumPy, Pandas |
| **Frontend** | HTML5, CSS3 (Custom UI), JavaScript |
| **Visualisasi Data** | Chart.js |
| **Environment** | Python `venv` |
| **Version Control** | Git |
| **OS Pengembangan** | Windows 11 |

---

## 📊 Model Performance

### 🫘 Modul 1 — Deteksi Batu Ginjal (YOLOv8)

Model YOLOv8 dilatih selama **50 epoch** untuk mendeteksi lokasi batu ginjal (`Tas_Var`) pada citra CT Scan secara real-time.

**Kurva mAP50 Selama Training:**

![Training mAP50 Curve](assets/kidney_map50_curve.png)

| Metrik | Nilai |
|---|---|
| **mAP@50 (Akhir)** | ~0.75 |
| **Epochs** | 50 |
| **Konvergensi** | Stabil mulai epoch ~15 |
| **Label Deteksi** | `Tas_Var` (Kidney Stone) |

> Model mencapai kestabilan performa mulai epoch ke-15 dengan mAP50 berkisar antara **0.70–0.76** hingga akhir training.

**Contoh Hasil Inferensi (CT Scan):**

![Kidney Detection Results](assets/kidney_detection_results.png)

> YOLOv8 berhasil melokalisasi batu ginjal dengan bounding box dan skor confidence pada berbagai orientasi CT Scan (confidence range: **0.64–0.71**).

---

### 🧠 Modul 2 — Klasifikasi Tumor Otak (MobileNetV2)

Model berbasis MobileNetV2 dengan Transfer Learning dilatih untuk mengklasifikasikan citra MRI otak ke dalam dua kelas: **Normal** dan **Tumor**.

**Hasil Evaluasi pada Data Validasi:**

| Metrik | Nilai |
|---|---|
| **Akurasi** | **90.00%** |
| **Loss** | 0.1744 |
| **Total Sampel Validasi** | 50 |

**Laporan Klasifikasi Detail:**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal (No Tumor) | 0.85 | 0.89 | 0.87 | 19 |
| Tumor (Yes) | 0.93 | 0.90 | **0.92** | 31 |
| **Overall Accuracy** | — | — | **0.90** | 50 |
| Macro Avg | 0.89 | 0.90 | 0.89 | 50 |
| Weighted Avg | 0.90 | 0.90 | 0.90 | 50 |

> ✅ Model menunjukkan **Precision 0.93** pada kelas Tumor, yang berarti tingkat false positive rendah — krusial dalam konteks medis untuk menghindari diagnosis positif palsu.

---

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/fiksdevploper/oncovision-ai.git
cd oncovision-ai
```

### 2. Buat & Aktifkan Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Akses Aplikasi

| Antarmuka | URL |
|---|---|
| **Dashboard Frontend** | `http://localhost:8000` |
| **Dokumentasi API (Swagger)** | `http://localhost:8000/docs` |
| **ReDoc** | `http://localhost:8000/redoc` |

---

## 📡 API Endpoints

### `POST /predict`
Menerima data tabular (CSV/JSON) untuk analisis risiko kanker payudara.

**Request Body:**
```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871]
}
```

**Response:**
```json
{
  "module": "breast_cancer",
  "prediction": 1,
  "label": "Benign",
  "confidence": 0.98,
  "risk_level": "Low",
  "timestamp": "2025-01-01T10:00:00Z"
}
```

---

### `POST /predict_visual`
Menerima input berupa **file citra medis** (JPEG/PNG) untuk deteksi tumor otak atau batu ginjal.

**Request:** `multipart/form-data`

| Field | Type | Deskripsi |
|---|---|---|
| `file` | `image/jpeg` atau `image/png` | File citra medis (MRI / USG / CT Scan) |
| `module` | `string` | `"brain_tumor"` atau `"kidney_stone"` |

**Response:**
```json
{
  "module": "kidney_stone",
  "detections": [
    {
      "label": "kidney_stone",
      "confidence": 0.91,
      "bounding_box": { "x": 120, "y": 85, "width": 45, "height": 40 }
    }
  ],
  "total_objects_detected": 1,
  "annotated_image_url": "/results/annotated_12345.png",
  "timestamp": "2025-01-01T10:00:00Z"
}
```

---

## 📦 Requirements

```
fastapi
uvicorn[standard]
ultralytics
tensorflow
opencv-python
numpy
pandas
scikit-learn
joblib
python-multipart
Pillow
```

> Install semua dependensi sekaligus dengan: `pip install -r requirements.txt`

---

## ⚕️ Disclaimer Medis

> **⚠️ PENTING — BACA SEBELUM MENGGUNAKAN**
>
> OncoVision AI adalah **alat bantu pendukung keputusan klinis** (*Clinical Decision Support Tool*), **bukan** alat diagnosis medis mandiri.
>
> - Hasil analisis dari sistem ini **tidak menggantikan** diagnosis, penilaian, atau rekomendasi dari dokter atau tenaga medis profesional yang berkualifikasi.
> - Setiap keputusan klinis tetap sepenuhnya menjadi tanggung jawab tenaga medis yang menangani pasien.
> - Model AI dalam sistem ini dilatih pada dataset tertentu dan memiliki keterbatasan performa pada kondisi klinis yang bervariasi.
> - Penggunaan sistem ini untuk tujuan diagnosis akhir tanpa supervisi medis **tidak dianjurkan** dan dapat menimbulkan risiko bagi pasien.
>
> Sistem ini dikembangkan untuk keperluan **penelitian, edukasi, dan pendukung skrining awal**. Selalu konsultasikan hasil dengan profesional kesehatan yang kompeten.

---

## 👤 Author

**Fikri** — ML Engineer Intern
[![GitHub](https://img.shields.io/badge/GitHub-fiksdevploper-181717?style=flat&logo=github)](https://github.com/fiksdevploper)
[![Portfolio](https://img.shields.io/badge/Portfolio-fiksdev.pages.dev-blue?style=flat&logo=cloudflare)](https://fiksdev.pages.dev)

---

## 📄 License

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ to support early disease detection in Indonesia
</p>
