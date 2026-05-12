# SmartPredict AI

SmartPredict AI is a machine learning-based predictive maintenance system designed to detect potential industrial machine failures before they occur.

---

## Project Objective

The goal of this project is to predict machine failure using sensor and operational data. The system helps reduce downtime, maintenance costs, and unexpected failures.

---

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.

Main features include:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type
- Machine failure target

---

## Machine Learning Problem

This is a binary classification problem:

- `0 = No Failure`
- `1 = Machine Failure`

---

## Key Challenge

The dataset is highly imbalanced:

- Most machines do not fail
- Failure cases are rare

Because of this, accuracy alone is not enough. Recall and F1-score are more important, especially for detecting failure cases.

---

## Feature Engineering

Two important engineered features were created.

### Temperature Difference

Process temperature minus air temperature.

This helps capture thermal stress on the machine.

### Power

Rotational speed multiplied by torque.

This helps represent mechanical load.

---

## Models Compared

Three models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

---

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Recall was considered especially important because missing a real failure can be costly in industrial environments.

---

## Hyperparameter Tuning

GridSearchCV was used to tune the XGBoost model.

The tuning process optimized recall because the main goal is to detect as many failures as possible.

---

## Explainability

SHAP was used to explain model predictions and understand which features influenced the model most.

---

## System Features

The system includes:

- Machine failure prediction
- Failure probability estimation
- Risk level classification
- Maintenance recommendations
- Critical maintenance work instructions
- AI-powered maintenance assistant
- FastAPI backend integration
- Streamlit interactive dashboard
- Downloadable maintenance reports

---

## Web Application

A Streamlit-based web application was developed to provide an interactive predictive maintenance platform. Users can enter machine operating parameters and receive:

- Failure prediction
- Failure probability
- Risk level classification
- Maintenance recommendations
- Critical work instructions
- AI-powered maintenance assistance

The system also integrates FastAPI as a backend service for model inference and Groq LLM APIs for the conversational maintenance assistant.

### Run the Application

Start FastAPI backend:
` uvicorn api.main:app --reload`

Start Streamlit frontend:
`streamlit run app/streamlit_app.py`


Installation

Clone the repository:
`git clone https://github.com/Wshaihr-Shaihr/SmartPredictAI.git
cd SmartPredictAI`

Install required dependencies:
`pip install -r requirements.txt`

## Project Structure

```text
smartpredict/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── xgboost_model.pkl
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   └── images/
├── src/
├── README.md
├── requirements.txt
└── .env
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- FastAPI
- Groq LLM API

---

## Conclusion

This project demonstrates how machine learning can support predictive maintenance in industrial environments.

By combining:

- Feature engineering
- Model tuning
- Explainability techniques
- FastAPI integration
- AI assistance
- Interactive Streamlit dashboards

the system can help detect machine failures before they occur and assist maintenance engineers in making faster and safer decisions.

Among all tested models, XGBoost achieved the best performance for failure detection, especially in terms of recall, making it suitable for predictive maintenance applications where missing a failure can be costly and dangerous.

---

## Developed By

Wassim Shairh  
AI & Machine Learning Engineer

- LinkedIn: https://www.linkedin.com/in/shaihr-eleid-wassim-384080123/
- GitHub: https://github.com/Wshaihr-Shaihr

Project: SmartPredict AI – Predictive Maintenance System
