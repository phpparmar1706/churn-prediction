import streamlit as st
import joblib
import pandas as pd
import os

st.title("Churn Prediction System")

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "backend", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "backend", "scaler.pkl"))

# User Inputs
CreditScore = st.number_input("CreditScore", min_value=0)
Age = st.number_input("Age", min_value=0)
Tenure = st.number_input("Tenure", min_value=0)
Balance = st.number_input("Balance", min_value=0.0)
NumOfProducts = st.number_input("NumOfProducts", min_value=0)
HasCrCard = st.number_input("HasCrCard", min_value=0, max_value=1)
IsActiveMember = st.number_input("IsActiveMember", min_value=0, max_value=1)
EstimatedSalary = st.number_input("EstimatedSalary", min_value=0.0)

# Predict Button
if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        [
            {
                "CreditScore": CreditScore,
                "Age": Age,
                "Tenure": Tenure,
                "Balance": Balance,
                "NumOfProducts": NumOfProducts,
                "HasCrCard": HasCrCard,
                "IsActiveMember": IsActiveMember,
                "EstimatedSalary": EstimatedSalary,
            }
        ]
    )

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)[0]

    # Probability
    probability = model.predict_proba(scaled_data)[0][1]

    # Result
    result = "Churn" if prediction == 1 else "Not Churn"

    st.subheader(f"Prediction: {result}")
    st.write(f"Churn Probability: {round(float(probability), 2)}")
