# -*- coding: utf-8 -*-
import os
import json
import argparse
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models.signature import infer_signature

class AmazonSalesTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_tracking()

    def setup_tracking(self):
        """Konfigurasi koneksi ke DagsHub."""
        if not os.getenv("GITHUB_ACTIONS"):
            dagshub.init(
                repo_owner='VanQ28', 
                repo_name='Model-Eksperimen_Richie-Leonard-Tjias', 
                mlflow=True
            )
        mlflow.set_tracking_uri("https://dagshub.com/VanQ28/Membangun_Model.mlflow")
        mlflow.set_experiment(self.config.experiment_name)

    def load_and_split(self):
        """Memuat dataset dan melakukan validasi kolom."""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Dataset missing: {self.config.data_path}")
        
        data = pd.read_csv(self.config.data_path)
        features = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']
        
        X = data[features].astype(float)
        y = data[self.config.target].astype(float)
        
        return train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )

    def execute_training(self):
        """Proses inti pelatihan dan logging."""
        X_train, X_test, y_train, y_test = self.load_and_split()
        
        # Inisialisasi Model
        regressor = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=self.config.random_state
        )
        
        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)
        
        # Evaluasi
        metrics = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2_score": r2_score(y_test, predictions)
        }

        # Persiapan Folder Output
        output_path = os.path.join(self.config.output_dir, "extras")
        os.makedirs(output_path, exist_ok=True)
        
        # Simpan Visualisasi
        self.generate_plots(y_test, predictions, output_path)
        
        # MLflow Logging
        run_id_env = os.environ.get("MLFLOW_RUN_ID")
        with (mlflow.start_run(run_id=run_id_env) if run_id_env else mlflow.start_run(run_name=self.config.run_name)):
            mlflow.log_params(vars(self.config))
            mlflow.log_metrics(metrics)
            
            # Advanced Artifacts (Penting untuk Kriteria 2)
            signature = infer_signature(X_train, regressor.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=regressor,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )
            
            # Log tambahan extras
            mlflow.log_artifacts(output_path, artifact_path="extras")
            
        print(f"Workflow Completed. R2 Score: {metrics['r2_score']:.4f}")

    def generate_plots(self, y_true, y_pred, folder):
        """Membuat plot evaluasi regresi."""
        plt.figure(figsize=(8, 5))
        plt.scatter(y_true, y_pred, color='blue', alpha=0.3)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        plt.title("Regression Performance: Actual vs Predicted")
        plt.savefig(os.path.join(folder, "regression_results.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Amazon_Preprocessing/amazon_train.csv")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--target", type=str, default="TotalAmount")
    parser.add_argument("--experiment_name", type=str, default="Amazon_Sales")
    parser.add_argument("--run_name", type=str, default="advanced_rf_training")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()
    
    trainer = AmazonSalesTrainer(args)
    trainer.execute_training()
