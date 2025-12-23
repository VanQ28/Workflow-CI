import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.ensemble import RandomForestRegressor
from mlflow.models.signature import infer_signature
from sklearn.utils import estimator_html_repr

# Inisialisasi DagsHub
dagshub.init(repo_owner='VanQ28', repo_name='Workflow_CI', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/VanQ28/Workflow_CI.mlflow")
mlflow.set_experiment("Amazon_Sales")

def train_model():
    base_path = os.path.dirname(__file__)
    path_train = os.path.join(base_path, "Amazon_Preprocessing", "amazon_train.csv")
    path_test = os.path.join(base_path, "Amazon_Preprocessing", "amazon_test.csv")
    
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    features = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']
    X_train = train[features].astype(float)
    y_train = train['TotalAmount'].astype(float)

    # Matikan log_models di autolog agar tidak duplikat dengan log manual kita
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="Random Forest Advanced Artifacts"):
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train) 
        
        # Signature & Input Example (Menghasilkan input_example.json)
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Estimator HTML
        html_path = "estimator.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(estimator_html_repr(model))
        mlflow.log_artifact(html_path, artifact_path="model")
        
        print("Training Selesai. Semua artefak advanced telah diunggah!")

if __name__ == "__main__":
    train_model()
