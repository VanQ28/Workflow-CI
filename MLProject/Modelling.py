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

def train_model(train_path, test_path, n_estimators, max_depth):
    # Load Data
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan. Pastikan path benar: {train_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_train = train['TotalAmount']
    X_test = test[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_test = test['TotalAmount']

    # MLFLOW
    with mlflow.start_run(run_name="Random Forest CI-Workflow"):
        # Log Parameter
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

        # Manual Logging Metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Feature Importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        importance_file = "feature_importance.csv"
        importance.to_csv(importance_file, index=False)
        mlflow.log_artifact(importance_file)

        # Summary Training
        summary_file = "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Model trained via CI Workflow.\n")
            f.write(f"R2 Score: {r2}\n")
            f.write(f"RMSE: {rmse}\n")
        mlflow.log_artifact(summary_file)

        # Log Model
        mlflow.sklearn.log_model(model, "random-forest-model")
        
        print(f"Retraining Selesai. R2: {r2}")

if __name__ == "__main__":
    # ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="Amazon_Preprocessing/amazon_train.csv")
    parser.add_argument("--test_path", type=str, default="Amazon_Preprocessing/amazon_test.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train_model(args.train_path, args.test_path, args.n_estimators, args.max_depth)
