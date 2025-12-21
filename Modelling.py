import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Inisialisasi DagsHub 
dagshub.init(repo_owner='VanQ28', repo_name='Membangun_Model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/VanQ28/Membangun_Model.mlflow")

def train_model():
    # Load data hasil preprocessing
    path_train = r"C:\Kuliah\DICODING ASAH\Project\Eksperimen_SML_Richie Leonard Tjias\Membangun Model\Amazon_Preprocessing\amazon_train.csv"
    path_test = r"C:\Kuliah\DICODING ASAH\Project\Eksperimen_SML_Richie Leonard Tjias\Membangun Model\Amazon_Preprocessing\amazon_test.csv"
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    X_train = train[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_train = train['TotalAmount']
    X_test = test[['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost']]
    y_test = test['TotalAmount']

    # Konfigurasi Parameter
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }

    with mlflow.start_run(run_name="Random Forest Baseline"):
        # Log Parameter Manual
        mlflow.log_params(params)

        # Training
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Prediksi & Metrik
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log Metrik Manual
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Simpan feature importance sebagai CSV
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        # Simpan Plot atau Log Summary (Contoh: teks ringkasan)
        with open("summary.txt", "w") as f:
            f.write(f"Model trained successfully with R2: {r2}")
        mlflow.log_artifact("summary.txt")

        # Log Model
        mlflow.sklearn.log_model(model, "random-forest-model")
        
        print(f"Model berhasil dilatih. R2: {r2}")

if __name__ == "__main__":
    train_model()