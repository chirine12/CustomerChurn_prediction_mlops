import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri('http://localhost:5000')


def prep_data(file_path):
    churn = pd.read_csv(file_path)

    numeric_columns = churn.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = churn.select_dtypes(include=["object", "category"]).columns.tolist()

    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        churn[col] = label_encoders[col].fit_transform(churn[col])

    def replace_outliers_with_nan(churn):
        for col in churn.select_dtypes(include=["int64", "float64"]).columns:
            Q1 = churn[col].quantile(0.25)
            Q3 = churn[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            churn[col] = churn[col].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
        return churn

    churn_with_nans = replace_outliers_with_nan(churn.copy())
    scaler = MinMaxScaler()
    churn_normalized = scaler.fit_transform(churn_with_nans[numeric_columns])
    knn_imputer = KNNImputer(n_neighbors=3)
    churn_imputed = knn_imputer.fit_transform(churn_normalized)

    churn_imputed_df = pd.DataFrame(churn_imputed, columns=numeric_columns)
    churn_imputed_df["Churn"] = churn["Churn"]

    X = churn_imputed_df.drop(columns=["Churn"])
    y = churn_imputed_df["Churn"]
    print("Shape de X:", X.shape)
    print("Colonnes de X:", X.columns)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    train_size = int(0.8 * len(churn))
    X_train_resampled, X_test = X_resampled[:train_size], X_resampled[train_size:]
    y_train_resampled, y_test = y_resampled[:train_size], y_resampled[train_size:]

    return X_train_resampled, y_train_resampled, X_test, y_test

def train_model(X_train_resampled, y_train_resampled, X_test, y_test):
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        try:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

            rf_model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
            grid_search.fit(X_train_resampled, y_train_resampled)

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_accuracy", grid_search.best_score_)
            mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")

            return grid_search.best_estimator_

        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"Erreur détectée : {e}")
            raise


def evaluate_model(model, X_test, y_test):	
    print(f"Type du modèle : {type(model)}")  # Ajout pour débogage
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    print("Matrice de confusion :")
    print(conf_matrix)
    print(f"Accuracy : {acc_score}")
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))

    # Log des métriques
    mlflow.log_metric("accuracy", acc_score)
    mlflow.log_metric("precision", clf_report["weighted avg"]["precision"])
    mlflow.log_metric("recall", clf_report["weighted avg"]["recall"])
    mlflow.log_metric("f1_score", clf_report["weighted avg"]["f1-score"])

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    model = joblib.load(filename)
    print(f"Type du modèle : {type(model)}")  # Pour vérifier le type
    return model


