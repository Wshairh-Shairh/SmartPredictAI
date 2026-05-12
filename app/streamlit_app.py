import streamlit as st
import joblib
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os
import requests

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(
    page_title="SmartPredict AI",
    page_icon="⚙️",
    layout="centered"
)
defaults = {
    "prediction_done": False,
    "prediction": 0,
    "probability": 0,
    "risk_level": "UNKNOWN",
    "recommendation": "No recommendation yet.",
    "work_instruction": "No work instructions available."
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value
# Load trained model
model = joblib.load("models/xgboost_model.pkl")

# App title
st.title("⚙️ SmartPredict AI")
st.sidebar.title("System Information")

st.sidebar.info("""
This AI system predicts industrial machine failures
using Machine Learning and Predictive Analytics.
""")

st.sidebar.success("Model: XGBoost")

st.sidebar.write("Version: 1.0")
st.caption("AI-Powered Predictive Maintenance Platform")

st.subheader("Predictive Maintenance System")

st.write("Enter machine parameters to predict failure risk.")

# User inputs
air_temp = st.number_input("Air Temperature [K]", value=300.0)

process_temp = st.number_input("Process Temperature [K]", value=310.0)

rotational_speed = st.number_input("Rotational Speed [rpm]", value=1500)

torque = st.number_input("Torque [Nm]", value=40.0)

tool_wear = st.number_input("Tool Wear [min]", value=5)

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

# One-hot encoding logic
type_l = 1 if machine_type == "L" else 0
type_m = 1 if machine_type == "M" else 0

# Feature engineering
temperature_difference = process_temp - air_temp

power = rotational_speed * torque

# Create dataframe
input_data = pd.DataFrame({
    "Air temperature K": [air_temp],
    "Process temperature K": [process_temp],
    "Rotational speed rpm": [rotational_speed],
    "Torque Nm": [torque],
    "Tool wear min": [tool_wear],
    "Temperature Difference": [temperature_difference],
    "Power": [power],
    "Type_L": [type_l],
    "Type_M": [type_m]
})

# call API to get prediction and probability
if st.button("Predict Failure"):

    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "air_temp": air_temp,
            "process_temp": process_temp,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
            "machine_type": machine_type
        }
    )

    result = response.json()

    prediction = result["prediction_value"]
    probability = result["failure_probability"]
    risk_level = result["risk_level"]
    recommendation = result["recommendation"]
    work_instruction = "\n".join(result["work_instructions"])

    # Save latest prediction to session_state
    st.session_state.prediction_done = True
    st.session_state.prediction = prediction
    st.session_state.probability = probability
    st.session_state.risk_level = risk_level
    st.session_state.recommendation = recommendation
    st.session_state.work_instruction = work_instruction


# Always display latest prediction after rerun
if st.session_state.prediction_done:

    if st.session_state.prediction == 1:
        st.error("⚠️ High Risk of Machine Failure!")
    else:
        st.success("✅ Machine Operating Normally")

    st.metric("Failure Risk", st.session_state.risk_level)

    if st.session_state.risk_level == "CRITICAL":
        st.warning("Immediate maintenance recommended.")
    else:
        st.info(st.session_state.recommendation)

    st.metric(
        label="Failure Probability",
        value=f"{st.session_state.probability*100:.2f}%"
    )

    if st.session_state.risk_level == "CRITICAL":
        st.subheader("🛠️ Work Instructions")
        st.markdown(st.session_state.work_instruction)

    report = f"""
SmartPredict AI - Maintenance Report

Prediction: {"FAILURE" if st.session_state.prediction == 1 else "NORMAL"}

Failure Probability: {st.session_state.probability*100:.2f}%

Risk Level: {st.session_state.risk_level}

Air Temperature: {air_temp}
Process Temperature: {process_temp}
Rotational Speed: {rotational_speed}
Torque: {torque}
Tool Wear: {tool_wear}
Machine Type: {machine_type}

Recommendation:
{st.session_state.recommendation}
"""

    st.download_button(
        label="Download Maintenance Report",
        data=report,
        file_name="maintenance_report.txt",
        mime="text/plain"
    )

    st.divider()

st.subheader("🛠 AI Maintenance Assistant")
if st.session_state.risk_level == "CRITICAL":

    st.warning(
        "⚠️ Critical condition detected. I can help you diagnose the issue, "
        "prioritize inspection steps, and explain how to reduce the failure risk."
    )

    with st.chat_message("assistant"):
        st.write(
            "The machine is currently in a CRITICAL condition. "
            "You can ask me: 'What should I inspect first?', "
            "'Why is this machine critical?', or "
            "'How can I reduce the risk?'"
        )
user_question = st.chat_input(
    "Ask about machine maintenance..."
)

if user_question:

    with st.chat_message("user"):
        st.write(user_question)

    system_prompt = f"""
    You are an industrial maintenance AI assistant.

    Current machine status:
    Risk Level: 
    {st.session_state.risk_level}


    Failure Probability: 
    {st.session_state.probability*100:.2f}%

    Recommendation:
    {st.session_state.recommendation}

    Work Instructions:
    {st.session_state.work_instruction}

    Explain clearly for maintenance engineers.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
    )

    ai_reply = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.write(ai_reply)