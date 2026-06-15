from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd

app = FastAPI()
handler = Mangum(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    preprocessor = joblib.load(os.path.join(BASE_DIR, 'preprocessor.pkl'))
    model = joblib.load(os.path.join(BASE_DIR, 'best_model_linear_regression.pkl'))
except Exception as e:
    preprocessor = None
    model = None
    print(f"Model loading error: {e}")

# Exact column order preprocessor was fitted on
NUM_COLS = ['Hours_Studied', 'Attendance(%)', 'Previous_Scores', 'Assignment_Score']
CAT_COLS = ['Parental_Education_Level']
ALL_COLS = ['Hours_Studied', 'Attendance(%)', 'Previous_Scores',
            'Parental_Education_Level', 'Assignment_Score']


class PredictRequest(BaseModel):
    Hours_Studied: float
    Attendance_pct: float               # mapped → Attendance(%) (invalid Python identifier)
    Previous_Scores: float
    Assignment_Score: float
    Parental_Education_Level: str = "unknown"  # "High School" | "College" | "Postgraduate" | "unknown"


@app.get("/")
def read_root():
    return {
        "message": "Student Performance Predictor API is Live!",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if preprocessor is None or model is None:
        return {"error": "Model not loaded"}
    try:
        input_dict = request.dict()

        # Rename to match exact training column name
        input_dict["Attendance(%)"] = input_dict.pop("Attendance_pct")

        # Build DataFrame in the exact column order preprocessor expects
        input_df = pd.DataFrame([input_dict])[ALL_COLS]

        # Step 1: preprocess with standalone preprocessor.pkl
        transformed = preprocessor.transform(input_df)

        # Step 2: predict using only the lr step from the pipeline
        lr = model.named_steps['lr']
        prediction = lr.predict(transformed)

        score = float(np.clip(round(float(prediction[0]), 2), 0, 100))

        return {
            "prediction": score,
            "grade": _grade(score)
        }
    except Exception as e:
        return {"error": str(e)}


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"