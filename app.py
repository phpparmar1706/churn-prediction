from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import asyncio

# Create FastAPI App
app = FastAPI()


# Load Model
model = joblib.load("model.pkl")

# Load Scaler
scaler = joblib.load("scaler.pkl")


# Request Body Schema

class ChurnInput(BaseModel):
        CreditScore: int
        Age: int
        Tenure: int
        Balance: float
        NumOfProducts: int
        HasCrCard: int 
        IsActiveMember: int
        EstimatedSalary: float



# Home Route
@app.get("/")
async def home():
    return {"message": "Churn Prediction API Running"}


# Prediction Route
@app.post("/predict")
async def predict_churn(data: ChurnInput):

    # Simulate async work
    await asyncio.sleep(0.1)

    # Convert to DataFrame
    input_data = pd.DataFrame(
        [
            {
          
                 'CreditScore': data.CreditScore,
                    'Age': data.Age,
                    'Tenure': data.Tenure,
                    'Balance': data.Balance,
                    'NumOfProducts': data.NumOfProducts,
                    'HasCrCard': data.HasCrCard,
                    'IsActiveMember': data.IsActiveMember,
                    'EstimatedSalary': data.EstimatedSalary
            }
        ]
    )

    # Scale Data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    # Probability
    probability = model.predict_proba(scaled_data)[0][1]

    # Final Result
    result = "Churn" if prediction == 1 else "Not Churn"

    return {"churn_prediction": result, "churn_probability": round(float(probability), 2)}
