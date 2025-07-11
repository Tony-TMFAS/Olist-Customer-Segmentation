from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

# Input model
class Customer(BaseModel):
    recency: float
    frequency: float
    monetary: float

@app.post("/predict")
def predict_segment(customer: Customer):
    data = np.array([[customer.recency, customer.frequency, customer.monetary]])
    scaled = scaler.transform(data)
    cluster = kmeans.predict(scaled)[0]
    return {"segment": int(cluster)}
