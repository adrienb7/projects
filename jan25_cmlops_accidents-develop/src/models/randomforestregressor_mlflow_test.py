# Imports librairies
import mlflow
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Expérience
mlflow.set_experiment("Accidents_Models")

# Données
data = pd.read_csv("data/preprocessed/df_accident_clean.csv")
#data['hrmn'] = data['hrmn'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
#index_with_nan = data.index[data.isnull().any(axis=1)]
#print(index_with_nan)
#data.drop(index_with_nan, 0, inplace=True)
data = data.dropna()
#
X = data.drop(['grav'], axis=1).astype('float')
y = data['grav']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
rf = RandomForestRegressor(**params)
rf.fit(X_train, y_train)

# Évaluation
y_pred = rf.predict(X_val)
metrics = {
    "mae": mean_absolute_error(y_val, y_pred),
    "mse": mean_squared_error(y_val, y_pred),
    "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
    "r2": r2_score(y_val, y_pred),
}
print(metrics)

# Log dans MLflow
print("=> MLFLOW")
with mlflow.start_run(run_name="First_run") as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        sk_model=rf,
        input_example=X_val.iloc[[0]],
        artifact_path="rfr_model"
    )