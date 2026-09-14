import joblib
import pandas as pd


model = joblib.load("/Users/shyamchauhan/Desktop/home/codes/fraud-detection/saved_models/fraud_model.pkl")


def predict_fraud(transaction):

    df = pd.DataFrame([transaction])

    probability = model.predict_proba(df)[0][1]

    risk_score = probability * 100

    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "fraud_probability": probability,
        "risk_score": risk_score,
        "risk_level": risk_level
    }