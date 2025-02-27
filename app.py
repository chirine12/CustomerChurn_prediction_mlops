from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

# Charger le modèle
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

@app.post("/predict/")
async def predict(data: dict):
    try:
        features = np.array(data["features"]).reshape(1, -1)  # Assurez-vous que les données sont de la bonne forme
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {e}")
