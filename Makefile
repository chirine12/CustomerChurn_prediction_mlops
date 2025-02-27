# =====================
# Configuration
# =====================
PYTHON = python3
VENV = venv
ACTIVATE_VENV = source $(VENV)/bin/activate
PROJECT_DIR = $(shell pwd)
REQUIREMENTS = requirements.txt

MODEL_PIPELINE = model_pipeline.py
MAIN_SCRIPT = main.py
MLFLOW_HOST = 0.0.0.0
MLFLOW_PORT = 5000
BACKEND_URI = sqlite:///mlflow.db

# =====================
# Installation et setup
# =====================

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(ACTIVATE_VENV) && pip install -r $(REQUIREMENTS)

# =====================
# Préparation et entraînement du modèle
# =====================

prep_data:
	$(VENV)/bin/python -c "from model_pipeline import prep_data; prep_data('$(PROJECT_DIR)/merged_churn.csv')"

train_model:
	$(VENV)/bin/python -c "from model_pipeline import prep_data, train_model; \
	X_train_resampled, y_train_resampled, X_test, y_test = prep_data('$(PROJECT_DIR)/merged_churn.csv'); \
	train_model(X_train_resampled, y_train_resampled, X_test, y_test)"

save_model:
	$(VENV)/bin/python -c "from model_pipeline import save_model, train_model, prep_data; \
	X_train_resampled, y_train_resampled, X_test, y_test = prep_data('$(PROJECT_DIR)/merged_churn.csv'); \
	model = train_model(X_train_resampled, y_train_resampled, X_test, y_test); \
	save_model(model, 'model.joblib')"

evaluate_model:
	$(VENV)/bin/python -c "from model_pipeline import load_model, prep_data, evaluate_model; \
	model = load_model('model.joblib'); \
	_, _, X_test, y_test = prep_data('$(PROJECT_DIR)/merged_churn.csv'); \
	evaluate_model(model, X_test, y_test)"

load_model:
	$(VENV)/bin/python -c "from model_pipeline import load_model, prep_data; \
	model = load_model('model.joblib'); \
	X_train, y_train, X_test, y_test = prep_data('$(PROJECT_DIR)/merged_churn.csv'); \
	print('Done loading the model!')"

# =====================
# API et Exécution
# =====================

run:
	$(VENV)/bin/python $(MAIN_SCRIPT)

run_api:
	$(VENV)/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8000

# =====================
# MLflow
# =====================

start-mlflow:
	$(VENV)/bin/mlflow ui --backend-store-uri $(BACKEND_URI) --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) &

stop-mlflow:
	pkill -f "mlflow ui --backend-store-uri $(BACKEND_URI)"

# =====================
# Docker
# =====================

DOCKER_IMAGE = shyrine_ourari_4ds5_mlops
DOCKER_USERNAME = shyrine12
DOCKER_TAG = latest
DOCKER_REPO = $(DOCKER_USERNAME)/$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-build:
	docker build -t $(DOCKER_REPO) .

docker-run:
	docker run -d -p 8080:80 --name $(DOCKER_IMAGE) $(DOCKER_REPO)

docker-push:
	docker push $(DOCKER_REPO)

docker-clean:
	docker rm -f $(DOCKER_IMAGE)
	docker rmi $(DOCKER_REPO)

# =====================
# Nettoyage
# =====================

clean:
	rm -rf $(VENV) mlruns mlflow.db __pycache__

# =====================
# Aide
# =====================

help:
	@echo "Usage des commandes Makefile :"
	@echo "  make install        - Installer les dépendances"
	@echo "  make prep_data      - Préparer les données"
	@echo "  make train_model    - Entraîner le modèle"
	@echo "  make save_model     - Sauvegarder le modèle"
	@echo "  make evaluate_model - Évaluer le modèle"
	@echo "  make load_model     - Charger et afficher le modèle"
	@echo "  make run            - Exécuter le pipeline complet"
	@echo "  make run_api        - Lancer l'API avec FastAPI"
	@echo "  make start-mlflow   - Lancer l'interface MLflow"
	@echo "  make stop-mlflow    - Arrêter MLflow"
	@echo "  make docker-build   - Construire l’image Docker"
	@echo "  make docker-run     - Lancer le conteneur Docker"
	@echo "  make docker-push    - Pousser l’image sur Docker Hub"
	@echo "  make docker-clean   - Supprimer les conteneurs et images Docker"
	@echo "  make clean          - Nettoyer l’environnement et les fichiers temporaires"

