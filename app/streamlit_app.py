import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="SmartPredict AI",
    page_icon="⚙️",
    layout="centered"
)
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

# Prediction button
# Prediction button
if st.button("Predict Failure"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:

        st.error("⚠️ High Risk of Machine Failure!")

        if probability > 0.8:

            risk_level = "CRITICAL"
            recommendation = "Immediate maintenance recommended."

        elif probability > 0.5:

            risk_level = "MEDIUM"
            recommendation = "Schedule maintenance soon."

        else:

            risk_level = "LOW"
            recommendation = "Monitor machine carefully."

        st.metric("Failure Risk", risk_level)
        st.warning(recommendation)

    else:

        risk_level = "LOW"
        recommendation = "No immediate maintenance required."

        st.success("✅ Machine Operating Normally")
        st.metric("Failure Risk", risk_level)
        st.info(recommendation)

    st.metric(
        label="Failure Probability",
        value=f"{probability*100:.2f}%"
    )

    report = f"""
SmartPredict AI - Maintenance Report

Prediction: {"FAILURE" if prediction == 1 else "NORMAL"}

Failure Probability: {probability*100:.2f}%

Risk Level: {risk_level}

Air Temperature: {air_temp}
Process Temperature: {process_temp}
Rotational Speed: {rotational_speed}
Torque: {torque}
Tool Wear: {tool_wear}
Machine Type: {machine_type}

Recommendation:
{recommendation}
"""

    st.download_button(
        label="Download Maintenance Report",
        data=report,
        file_name="maintenance_report.txt",
        mime="text/plain"
    )