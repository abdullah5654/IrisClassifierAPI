from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from model_utils import train_and_save_model, predict_species, MODEL_PATH
import os

# Train model once if not already saved
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

app = FastAPI(title="ML Model API", description="Iris Flower Prediction API")


# Input schema using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok", "message": "Server is running"}


@app.post("/predict")
def predict(data: IrisFeatures):
    try:
        features = [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]
        prediction = predict_species(features)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
