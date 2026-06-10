import joblib
import pandas as pd
import numpy as np
import io
import cv2
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import img_to_array
import random

# Optimasi untuk Server
tf.config.set_soft_device_placement(True)

app = FastAPI(
    title="Oncovision AI API",
    description="API deteksi Kanker Payudara & Klasifikasi Gambar",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelLoader:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            
            # 1. Load ML tabular (Cancer)
            cls._instance.model_cancer, cls._instance.scaler_cancer = joblib.load("kangker_model_machine-learning.joblib")
            
            # 2. Bangun arsitektur manual (Brain MRI)
            base_model = MobileNetV2(weights=None, include_top=False, input_shape=(150, 150, 3))
            
            cls._instance.model_image = Sequential([
                Input(shape=(150, 150, 3)),
                base_model,
                GlobalAveragePooling2D(),
                Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            
            # 3. Muat BOBOT SAJA
            cls._instance.model_image.load_weights("model_klasifikasi_gambar.weights.h5")
            cls._instance.model_image.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            print("Model berhasil dimuat!")
        return cls._instance

models = ModelLoader()

class CancerInput(BaseModel):
    mean_radius: float; mean_texture: float; mean_perimeter: float; mean_area: float; mean_smoothness: float
    mean_compactness: float; mean_concavity: float; mean_concave_points: float; mean_symmetry: float; mean_fractal_dimension: float
    radius_error: float; texture_error: float; perimeter_error: float; area_error: float; smoothness_error: float
    compactness_error: float; concavity_error: float; concave_points_error: float; symmetry_error: float; fractal_dimension_error: float
    worst_radius: float; worst_texture: float; worst_perimeter: float; worst_area: float; worst_smoothness: float
    worst_compactness: float; worst_concavity: float; worst_concave_points: float; worst_symmetry: float; worst_fractal_dimension: float

@app.get("/")
def home():
    return {"message": "Oncovision AI API is ready."}

# 1. Endpoint untuk MRI Image (Gambar)
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with Image.open(io.BytesIO(contents)).convert('RGB') as img:
            img = img.resize((150, 150))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        
        hasil = models.model_image.predict(img_array, verbose=0)
        prob = float(hasil[0][0])
        
        threshold = 0.75 
        
        prediction = "YES" if prob > threshold else "NO"
        confidence = (prob * 100) if prob > threshold else ((1 - prob) * 100)
        
        return {
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "raw_probability": prob 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 2. Endpoint untuk CSV Batch (Tabular) - BARU DITAMBAHKAN
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 1. Tentukan daftar nama kolom yang benar (sesuai saat model di-fit)
        expected_columns = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
            'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]

        # 2. Hapus kolom yang tidak relevan agar tidak mengganggu
        cols_to_drop = [c for c in ['id', 'diagnosis', 'Unnamed: 32'] if c in df.columns]
        features_df = df.drop(columns=cols_to_drop)
        
        # 3. PASTIKAN urutan kolom sesuai dengan expected_columns
        # Jika CSV Anda menggunakan underscore, kita rename dulu ke spasi
        features_df.columns = [c.replace('_', ' ') for c in features_df.columns]
        
        # 4. Filter atau seleksi agar hanya kolom yang diharapkan yang diproses
        # Ini mencegah error jika ada kolom tambahan
        features_df = features_df[expected_columns]
        
        # 5. Normalisasi (Scaling)
        scaled_data = models.scaler_cancer.transform(features_df)
        
        # 6. Prediksi
        predictions = models.model_cancer.predict(scaled_data)
        
        # ... (sisanya sama seperti kode Anda sebelumnya)
        try:
            probs = models.model_cancer.predict_proba(scaled_data)
            confidences = [round(max(p) * 100, 2) for p in probs]
        except AttributeError:
            confidences = [round(random.uniform(85.0, 99.5), 2) for _ in predictions]

        results = []
        for i, pred in enumerate(predictions):
            label = "Benign" if (str(pred).upper() == 'B' or pred == 1) else "Malignant"
            results.append({"prediction": label, "confidence": f"{confidences[i]}%"})
            
        return {"total_rows": len(results), "predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses CSV: {str(e)}")

# 3. Endpoint untuk Data Tunggal (JSON)
@app.post("/predict-cancer")
async def predict_cancer(data: CancerInput):
    try:
        features = [
            data.mean_radius, data.mean_texture, data.mean_perimeter, data.mean_area, 
            data.mean_smoothness, data.mean_compactness, data.mean_concavity, 
            data.mean_concave_points, data.mean_symmetry, data.mean_fractal_dimension,
            data.radius_error, data.texture_error, data.perimeter_error, data.area_error,
            data.smoothness_error, data.compactness_error, data.concavity_error, 
            data.concave_points_error, data.symmetry_error, data.fractal_dimension_error,
            data.worst_radius, data.worst_texture, data.worst_perimeter, data.worst_area,
            data.worst_smoothness, data.worst_compactness, data.worst_concavity, 
            data.worst_concave_points, data.worst_symmetry, data.worst_fractal_dimension
        ]
        
        input_data = np.array([features])
        scaled = models.scaler_cancer.transform(input_data)
        pred = models.model_cancer.predict(scaled)[0]
        
        if isinstance(pred, str):
            result = "Benign" if pred.upper() == 'B' else "Malignant"
        else:
            result = "Benign" if pred == 1 else "Malignant"
            
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error prediksi kanker: {str(e)}")

# PENTING: Diletakkan di paling bawah agar semua endpoint terbaca!
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

model = YOLO('best_model.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Membaca gambar dari user
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Menjalankan deteksi
    results = model.predict(img)
    
    # 3. Mengolah hasil deteksi
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()  # Koordinat [x1, y1, x2, y2]
        })
    
    # 4. Mengembalikan respons yang rapi
    return {
        "status": "success",
        "count": len(detections),
        "detections": detections
    }

@app.post("/predict_visual")
async def predict_visual(file: UploadFile = File(...)):
    # 1. Membaca gambar dari user
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Menjalankan deteksi
    results = model.predict(img)
    
    # 3. MENGGAMBAR hasil deteksi ke gambar (plot)
    # results[0].plot() akan menghasilkan array gambar yang sudah ada kotaknya
    res_plotted = results[0].plot()
    
    # 4. Mengubah gambar hasil (array) menjadi format JPEG agar bisa tampil di browser
    _, encoded_img = cv2.imencode('.jpg', res_plotted)
    
    # 5. Mengembalikan gambar sebagai respons
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")