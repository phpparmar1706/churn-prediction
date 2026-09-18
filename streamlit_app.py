import streamlit as st
import requests

st.title("Churn Prediction System")


# User Inputs
CreditScore = st.number_input("CreditScore", min_value=0)

Age = st.number_input("Age")

Tenure = st.number_input("Tenure")

Balance = st.number_input("Balance")

NumOfProducts = st.number_input("NumOfProducts")

HasCrCard = st.number_input("HasCrCard")

IsActiveMember = st.number_input("IsActiveMember")

EstimatedSalary = st.number_input("EstimatedSalary")

# Predict Button
if st.button("Predict Churn"):

    payload = {
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
       
    }

    # API Call
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    result = response.json()

st.subheader(f"Prediction: {result['churn_prediction']}")

st.write(f"Churn Probability: {result['churn_probability']}")
