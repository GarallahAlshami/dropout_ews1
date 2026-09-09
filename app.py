from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="Dropout Early Warning System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")
    lr_model = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
except Exception:
    xgb_model = lr_model = None
    feature_cols = []

class Student(BaseModel):
    attendance_rate: float
    average_grade: float
    failed_courses: int = 0
    assignments_missing: int = 0
    financial_risk: int = 0
    academic_warning: int = 0
    previous_dropout_risk: int = 0

@app.get("/")
def root():
    return {"name": "Dropout Early Warning System", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": xgb_model is not None and lr_model is not None}

@app.get("/students")
def students():
    path = DATA_DIR / "students.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Student data not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

def make_features(student: Student):
    values = student.model_dump()
    return pd.DataFrame([values]).reindex(columns=feature_cols, fill_value=0)

@app.post("/predict")
def predict(student: Student):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")
    X = make_features(student)
    prob = float(xgb_model.predict_proba(X)[0][1])
    risk = "High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low"
    return {"dropout_probability": round(prob, 4), "risk_level": risk}

@app.get("/metrics")
def metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    return pd.read_json(path, typ="series").to_dict()
