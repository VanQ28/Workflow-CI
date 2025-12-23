import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import argparse 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD")
dagshub_username = os.getenv("MLFLOW_TRACKING_USERNAME")
repo_name = "Model-Eksperimen_Richie-Leonard-Tjias"

if dagshub_token and dagshub_username:
    os.environ['DAGSHUB_USER_TOKEN'] = dagshub_token
    dagshub.init(repo_owner=dagshub_username, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_username}/{repo_name}.mlflow")
else:
    dagshub.init(repo_owner='VanQ28', repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/VanQ28/{repo_name}.mlflow")

mlflow.set_experiment("Amazon_Sales")

def train_model(train_path, test_path, n_estimators, max_depth):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_train_path = os.path.join(base_dir, train_path)
    full_test_path = os.path.join(base_dir, test_path)

    if not os.path.exists(full_train_path) or not os.path.exists(full_test_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan! Cek path: {full_train_path}")

    # Load Data
    train = pd.read_csv(full_train_path)
    test = pd.read_csv(full_test_path)

    X_train = train[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_train = train['TotalAmount']
    X_test = test[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_test = test['TotalAmount']

    # MLflow Run
    with mlflow.start_run(run_name="Random Forest CI-Workflow"):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42
        }
        mlflow.log_params(params)

        # Model Training
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluation
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Manual Logging Metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Artefak 1: Feature Importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        # Artefak 2: Summary Text
        with open("summary.txt", "w") as f:
            f.write(f"Model trained via CI Workflow.\n")
            f.write(f"R2 Score: {r2}\n")
            f.write(f"RMSE: {rmse}\n")
        mlflow.log_artifact("summary.txt")

        # Log Model
        mlflow.sklearn.log_model(model, "random-forest-model")
        
        print(f"Retraining Berhasil! R2: {r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="Amazon_Preprocessing/amazon_train.csv")
    parser.add_argument("--test_path", type=str, default="Amazon_Preprocessing/amazon_test.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train_model(args.train_path, args.test_path, args.n_estimators, args.max_depth)
