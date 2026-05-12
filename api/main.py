from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="SmartPredict AI API",
    description="Predictive Maintenance API using XGBoost",
    version="1.0"
)


class MachineInput(BaseModel):
    air_temp: float
    process_temp: float
    rotational_speed: int
    torque: float
    tool_wear: int
    machine_type: str


@app.get("/")
def home():
    return {"message": "SmartPredict AI API is running"}


@app.post("/predict")
def predict_failure(data: MachineInput):

    type_l = 1 if data.machine_type == "L" else 0
    type_m = 1 if data.machine_type == "M" else 0

    temperature_difference = data.process_temp - data.air_temp
    power = data.rotational_speed * data.torque

    input_data = pd.DataFrame({
        "Air temperature K": [data.air_temp],
        "Process temperature K": [data.process_temp],
        "Rotational speed rpm": [data.rotational_speed],
        "Torque Nm": [data.torque],
        "Tool wear min": [data.tool_wear],
        "Temperature Difference": [temperature_difference],
        "Power": [power],
        "Type_L": [type_l],
        "Type_M": [type_m]
    })

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    if prediction == 1:
        if probability > 0.8:
            risk_level = "CRITICAL"
            recommendation = "Immediate maintenance recommended."
            work_instructions = [
                "Stop the machine safely.",
                "Inspect tool wear and replace the tool if needed.",
                "Check torque load and mechanical resistance.",
                "Inspect rotational speed abnormalities.",
                "Check cooling and temperature conditions.",
                "Restart only after inspection and monitor readings."
            ]
        elif probability > 0.5:
            risk_level = "MEDIUM"
            recommendation = "Schedule maintenance soon."
            work_instructions = [
                "Plan inspection during the next maintenance window.",
                "Monitor tool wear and torque behavior.",
                "Check temperature trend over time."
            ]
        else:
            risk_level = "LOW"
            recommendation = "Monitor machine carefully."
            work_instructions = [
                "Continue monitoring sensor values.",
                "Review readings if the risk increases."
            ]
    else:
        risk_level = "LOW"
        recommendation = "No immediate maintenance required."
        work_instructions = [
            "Continue normal operation.",
            "Keep monitoring machine parameters."
        ]

    return {
        "prediction": "FAILURE" if prediction == 1 else "NORMAL",
        "prediction_value": prediction,
        "failure_probability": round(probability, 4),
        "failure_probability_percent": round(probability * 100, 2),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "work_instructions": work_instructions
    }