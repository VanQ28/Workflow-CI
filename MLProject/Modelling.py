import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models.signature import infer_signature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Amazon_Preprocessing/amazon_train.csv")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--target", type=str, default="TotalAmount")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default="Amazon_Sales")
    parser.add_argument("--run_name", type=str, default="local_run")
    args = parser.parse_args()

    # 1. Load Data
    df = pd.read_csv(args.data_path)
    features = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']
    X = df[features].astype(float)
    y = df[args.target].astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 2. Training
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=args.random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # 3. Metrics
    metrics = {
        "mse": float(mean_squared_error(y_test, preds)),
        "r2_score": float(r2_score(y_test, preds))
    }

    # 4. Extras (Plot)
    extras_path = os.path.join(args.output_dir, "extras")
    os.makedirs(extras_path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.4)
    plt.savefig(os.path.join(extras_path, "training_results.png"))
    plt.close()

    
    mlflow.log_params(vars(args))
    mlflow.log_metrics(metrics)
    
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5]
    )
    mlflow.log_artifacts(extras_path, artifact_path="extras")
    
    print(f"Selesai! R2: {metrics['r2_score']}")

if __name__ == "__main__":
    main()
