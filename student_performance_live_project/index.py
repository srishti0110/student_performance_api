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
    model = joblib.load(os.path.join(BASE_DIR, 'best_model_random_forest.pkl'))
except Exception as e:
    model = None
    print(f"Model loading error: {e}")

# All columns the pipeline preprocessor expects (in training order)
NUM_COLS = [
    'Hours_Studied', 'Attendance(%)', 'Previous_Scores', 'Assignment_Score',
    'Sleep_Hours', 'Physical_Activity', 'Tutoring_Sessions'
]

CAT_COLS = [
    'Parental_Education_Level', 'Teacher_Quality', 'Peer_Influence',
    'Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities',
    'Motivation_Level', 'Family_Income', 'School_Type',
    'Distance_from_Home', 'Access_to_Resources'
]

ALL_COLS = NUM_COLS + CAT_COLS


class PredictRequest(BaseModel):
    # Numeric
    Hours_Studied: float
    Attendance_pct: float          # mapped → Attendance(%)
    Previous_Scores: float
    Assignment_Score: float
    Sleep_Hours: float = 7.0
    Physical_Activity: float = 3.0
    Tutoring_Sessions: int = 1

    # Categorical
    Parental_Education_Level: str = "unknown"
    Teacher_Quality: str = "Medium"
    Peer_Influence: str = "Neutral"
    Extracurricular_Activities: str = "No"
    Internet_Access: str = "Yes"
    Learning_Disabilities: str = "No"
    Motivation_Level: str = "Medium"
    Family_Income: str = "Medium"
    School_Type: str = "Public"
    Distance_from_Home: str = "Near"
    Access_to_Resources: str = "High"


@app.get("/")
def read_root():
    return {"message": "Student Performance Predictor API is Live!", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}
    try:
        input_dict = request.dict()

        # Rename Attendance_pct → Attendance(%) to match training column name
        input_dict["Attendance(%)"] = input_dict.pop("Attendance_pct")

        input_df = pd.DataFrame([input_dict])[ALL_COLS]

        prediction = model.predict(input_df)
        score = float(np.clip(round(float(prediction[0]), 2), 0, 100))

        return {
            "prediction": score,
            "grade": _grade(score)
        }
    except Exception as e:
        return {"error": str(e)}


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    return "F"