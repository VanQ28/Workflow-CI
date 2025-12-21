import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import argparse 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Inisialisasi DagsHub
dagshub.init(repo_owner='VanQ28', repo_name='Model-Eksperimen_Richie-Leonard-Tjias', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/VanQ28/Model-Eksperimen_Richie-Leonard-Tjias.mlflow")
mlflow.set_experiment("Amazon_Sales")

def train_model(n_estimators, max_depth):
    # Load Data
    base_path = os.path.dirname(__file__)
    path_train = os.path.join(base_path, "Amazon_Preprocessing", "amazon_train.csv")
    path_test = os.path.join(base_path, "Amazon_Preprocessing", "amazon_test.csv")
    
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    X_train = train[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_train = train['TotalAmount']
    X_test = test[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_test = test['TotalAmount']

    # 3. MLFLOW RUN
    with mlflow.start_run(run_name="Random Forest CI-Workflow"):
        # Log Parameter dari argparse
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42
        }
        mlflow.log_params(params)

        # Training
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Prediksi & Metrik
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Manual Logging Metrics (Kriteria Advanced)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Artefak 1: Feature Importance (Kriteria Advanced)
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        # Artefak 2: Summary (Kriteria Advanced)
        with open("summary.txt", "w") as f:
            f.write(f"Model trained via CI Workflow. R2 Score: {r2}")
        mlflow.log_artifact("summary.txt")

        # Log Model
        mlflow.sklearn.log_model(model, "random-forest-model")
        
        print(f"Retraining Selesai. R2: {r2}")

if __name__ == "__main__":
    # 4. ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth)