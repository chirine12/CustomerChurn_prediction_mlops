from fastapi import FastAPI
from model_pipeline import prep_data, train_model, evaluate_model, save_model, load_model

app = FastAPI()

# Chemin du dataset
DATASET_PATH = "/home/chirine_ourari/shyrine_ourari_4DS5_ml_project/merged_churn.csv"
MODEL_PATH = "model.joblib"

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de Machine Learning !"}

@app.post("/train")
def train():
    """
    Entraîne le modèle et le sauvegarde.
    """
    try:
        # Préparation des données
        X_train_resampled, y_train_resampled, X_test, y_test = prep_data(DATASET_PATH)

        # Entraînement du modèle
        model = train_model(X_train_resampled, y_train_resampled, X_test, y_test)

        # Évaluation du modèle
        evaluate_model(model, X_test, y_test)

        # Sauvegarde du modèle
        save_model(model, MODEL_PATH)

        return {"message": "Modèle entraîné et sauvegardé avec succès !"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/evaluate")
def evaluate():
    """
    Charge le modèle et l'évalue.
    """
    try:
        # Charger le modèle
        loaded_model = load_model(MODEL_PATH)

        # Préparation des données
        X_train_resampled, y_train_resampled, X_test, y_test = prep_data(DATASET_PATH)

        # Évaluation du modèle
        score = evaluate_model(loaded_model, X_test, y_test)

        return {"score": score}

    except Exception as e:
        return {"error": str(e)}

# Lancer l'API si exécuté directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

