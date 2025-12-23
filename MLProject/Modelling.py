# -*- coding: utf-8 -*-
import os
import json
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

class AmazonSalesLocal:
    def __init__(self, settings):
        self.settings = settings
        # Set tracking ke folder lokal di dalam workspace GitHub
        mlflow.set_tracking_uri(f"file:{os.getcwd()}/mlruns")
        mlflow.set_experiment(self.settings.experiment_name)

    def prepare_data(self):
        df = pd.read_csv(self.settings.data_path)
        features = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']
        X = df[features].astype(float)
        y = df[self.settings.target].astype(float)
        return train_test_split(X, y, test_size=self.settings.test_size, random_state=self.settings.random_state)

    def run_training(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.settings.random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        metrics = {
            "mse": mean_squared_error(y_test, preds),
            "r2_score": r2_score(y_test, preds)
        }

        # Simpan plot lokal
        extras_path = os.path.join(self.settings.output_dir, "extras")
        os.makedirs(extras_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, preds, alpha=0.4, color='blue')
        plt.savefig(os.path.join(extras_path, "training_results.png"))
        plt.close()
        
        # Log ke MLflow Lokal
        with mlflow.start_run(run_name=self.settings.run_name):
            mlflow.log_params(vars(self.settings))
            mlflow.log_metrics(metrics)
            
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )
            mlflow.log_artifacts(extras_path, artifact_path="extras")
            
        print(f"Training Lokal Selesai. R2: {metrics['r2_score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Amazon_Preprocessing/amazon_train.csv")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--target", type=str, default="TotalAmount")
    parser.add_argument("--experiment_name", type=str, default="Amazon_Local_Exp")
    parser.add_argument("--run_name", type=str, default="local_run")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    AmazonSalesLocal(args).run_training()
