import joblib
import pandas as pd
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Cancer Detection API",
    description="API untuk deteksi kanker payudara menggunakan model SVC (Malignant vs Benign)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model, scaler = joblib.load("kangker_model_machine-learning.joblib")
    print("Model dan Scaler berhasil dimuat.")
except Exception as e:
    print(f" Gagal memuat model: {e}")

class CancerInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

@app.get("/")
def home():
    return {"message": "Cancer Detection API is running. Go to /docs for documentation."}

@app.post("/predict")
async def predict_json(data: CancerInput):
    try:
        input_dict = data.dict()
        df_input = pd.DataFrame([input_dict])
        df_input.columns = [col.replace('_', ' ') for col in df_input.columns]
        input_scaled = scaler.transform(df_input)
        prediction = int(model.predict(input_scaled)[0])
        prob = model.predict_proba(input_scaled)[0]
        
        label = "Benign" if prediction == 1 else "Malignant"
        
        return {
            "prediction_code": prediction,
            "result": label,
            "confidence_score": {
                "malignant": round(prob[0] * 100, 2),
                "benign": round(prob[1] * 100, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File harus berformat .csv")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if df.shape[1] < 30:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV harus memiliki minimal 30 kolom fitur. Ditemukan: {df.shape[1]}"
            )
        features = df.iloc[:, :30] 
        features.columns = [col.replace('_', ' ') for col in features.columns]
        scaled_features = scaler.transform(features)
        preds = model.predict(scaled_features)
        probs = model.predict_proba(scaled_features)
        results = []
        
        for i in range(len(preds)):
            results.append({
                "row_index": i + 1,
                "prediction": "Benign" if preds[i] == 1 else "Malignant",
                "confidence": f"{round(max(probs[i]) * 100, 2)}%"
            })

        return {
            "filename": file.filename,
            "total_rows": len(results),
            "predictions": results
        }
    except Exception as e:
        print(f"Error processing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses file CSV: {str(e)}")