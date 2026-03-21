from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Iris Classifier API")

model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict/")
def predict_species(data: IrisInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    species = ["setosa", "versicolor", "virginica"]
    return {
        "predicted_class": int(prediction),
        "species": species[prediction]
    }

@app.get("/health")
def health():
    return {"status": "ok"}
