import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

dagshub_token = os.getenv("DAGSHUB_TOKEN")
repo_owner = "VanQ28"
repo_name = "Workflow_CI"

if dagshub_token:
    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

def train_model(args):
    # Setup Experiment
    mlflow.set_experiment(args.experiment_name)
    
    # Path dataset absolut
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_train_path = os.path.join(base_dir, args.train_path)
    full_test_path = os.path.join(base_dir, args.test_path)
    
    # Load Data
    train_df = pd.read_csv(full_train_path)
    test_df = pd.read_csv(full_test_path)
    
    features = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']
    X_train = train_df[features]
    y_train = train_df[args.target]
    X_test = test_df[features]
    y_test = test_df[args.target]

    # REVISI: Menggunakan nested=True agar tidak konflik dengan Run ID dari MLflow Project
    with mlflow.start_run(run_name=args.run_name, nested=True):
        mlflow.log_params(vars(args))
        
        model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Log Model ke folder 'model'
        mlflow.sklearn.log_model(model, "model")
        
        # Simpan artefak tambahan
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/summary.txt", "w") as f:
            f.write(f"R2: {r2}")
        mlflow.log_artifacts(args.output_dir, artifact_path="extras")
        
        print(f"CI Retraining Success! R2: {r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--random_state", type=int)
    args = parser.parse_args()
    train_model(args)
