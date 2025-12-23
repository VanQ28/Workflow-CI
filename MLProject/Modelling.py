import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

class AmazonSalesTrainer:
    def __init__(self, args):
        self.args = args

    def run(self):
        # Load Data
        df = pd.read_csv(self.args.data_path)
        features = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']
        X = df[features].astype(float)
        y = df[self.args.target].astype(float)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.args.test_size, random_state=self.args.random_state
        )

        # Set experiment
        mlflow.sklearn.autolog()

        # Training
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=self.args.random_state
        )
        
        # Proses fit akan memicu autolog untuk mencatat metadata model
        model.fit(X_train, y_train)
        
        # Predict & Evaluation
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)

        # Save Plot (Manual Artifact tetap diperlukan untuk Plot Kustom)
        os.makedirs(self.args.output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, preds, alpha=0.4)
        plt.title("Amazon Sales: Actual vs Predicted")
        plt.savefig(os.path.join(self.args.output_dir, "training_results.png"))
        plt.close()

        # Advanced Logging (Signature & Input Example)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        # Log plot kustom ke folder extras
        mlflow.log_artifacts(self.args.output_dir, artifact_path="extras")
        
        print(f"Training Selesai dengan Autolog. R2 Score: {r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Amazon_Preprocessing/amazon_train.csv")
    parser.add_argument("--target", type=str, default="TotalAmount")
    parser.add_argument("--output_dir", type=str, default="extras")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default="Amazon_Sales")
    parser.add_argument("--run_name", type=str, default="autolog_run")
    
    args = parser.parse_args()
    AmazonSalesTrainer(args).run()
