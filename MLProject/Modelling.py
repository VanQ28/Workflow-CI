import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

dagshub_token = os.getenv("DAGSHUB_TOKEN")
repo_owner = "VanQ28"
repo_name = "Workflow_CI" 

if dagshub_token:
    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
else:
    try:
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    except:
        pass

mlflow.set_experiment("Amazon_Sales")

def train_model(train_path, test_path, n_estimators, max_depth):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_train_path = os.path.join(base_dir, train_path)
    full_test_path = os.path.join(base_dir, test_path)

    if not os.path.exists(full_train_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {full_train_path}")

    train = pd.read_csv(full_train_path)
    test = pd.read_csv(full_test_path)

    X_train = train[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_train = train['TotalAmount']
    X_test = test[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_test = test['TotalAmount']

    with mlflow.start_run(run_name="CI-Retraining-Final"):
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
        mlflow.log_params(params)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Artefak
        importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
        importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        mlflow.sklearn.log_model(model, "random-forest-model")
        print(f"Training Selesai di GitHub Actions! R2: {r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="Amazon_Preprocessing/amazon_train.csv")
    parser.add_argument("--test_path", type=str, default="Amazon_Preprocessing/amazon_test.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train_model(args.train_path, args.test_path, args.n_estimators, args.max_depth)
