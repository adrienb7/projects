import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
from mlflow import MlflowClient
import mlflow
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Define tracking_uri
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
apple_experiment = mlflow.set_experiment("AI_Models")
run_name = "first_run"
artifact_path = "rf_ai"
apple_experiment = mlflow.set_experiment("RandomForestRegressor")
run_name = "First_run"
artifact_path = "rfr_model"

# Import Database
data = pd.read_csv("data/preprocessed/df_accidents_clean_test.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')
y = data["demand"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
# Charger les données
print("Chargement des données...")
start_time = time.time()
df = pd.read_csv("data\\df_accidents_nettoye_train.csv", delimiter=";", encoding="utf-8", quotechar='"', on_bad_lines='skip')
print(f"Chargement des données terminé en {time.time() - start_time:.2f} secondes")

# Séparer les caractéristiques (X) et la cible (y)
print("Séparation des caractéristiques et de la cible...")
start_time = time.time()
X = df.drop(columns=['grav'])  # Features
y = df['grav']  # Target
print(f"Séparation des caractéristiques et de la cible terminé en {time.time() - start_time:.2f} secondes")

# Convertir les colonnes catégorielles en numériques (si nécessaire)
print("Conversion des variables catégorielles en numériques...")
start_time = time.time()
X = pd.get_dummies(X, drop_first=True)
print(f"Conversion des variables catégorielles terminé en {time.time() - start_time:.2f} secondes")

# Diviser en ensembles d'entraînement et de test
print("Division des données en ensembles d'entraînement et de test...")
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Division des données terminé en {time.time() - start_time:.2f} secondes")

# Conversion explicite en numériques
X_train = X_train.apply(pd.to_numeric, errors='coerce')  # Remplacer les erreurs par NaN
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Remplir les NaN avec une valeur spécifique (par exemple, la moyenne de chaque colonne)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Convertir les DataFrames pandas en matrices DMatrix pour XGBoost
print("Conversion des données en DMatrix pour XGBoost...")
start_time = time.time()
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Convertir en numpy arrays
print("Conversion de X_train, X_test, y_train et y_test en arrays numpy...")
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Vérifier la forme des matrices
print(f"Forme de X_train après conversion en numpy : {X_train.shape}")
print(f"Forme de X_test après conversion en numpy : {X_test.shape}")
print(f"Forme de y_train après conversion en numpy : {y_train.shape}")
print(f"Forme de y_test après conversion en numpy : {y_test.shape}")

# Conversion en DMatrix pour XGBoost
print("Conversion en DMatrix pour XGBoost...")
try:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print("Conversion en DMatrix réussie.")
except Exception as e:
    print(f"Erreur lors de la conversion en DMatrix : {e}")

print(f"Conversion en DMatrix terminé en {time.time() - start_time:.2f} secondes")

# Définir les paramètres pour utiliser GPU
print("Définition des paramètres pour XGBoost avec GPU...")
start_time = time.time()
params = {
    'objective': 'multi:softmax',  # Classification multiclasse
    'num_class': len(y.unique()),  # Nombre de classes uniques
    'tree_method': 'hist', 
    'device': 'cuda', # Utiliser le GPU pour l'entraînement
    'predictor': 'gpu_predictor',  # Utiliser le GPU pour la prédiction
    'eval_metric': 'merror',  # Utiliser le taux d'erreur pour évaluation
    'max_depth': 6,  # Profondeur maximale de l'arbre
    'learning_rate': 0.01,  # Taux d'apprentissage
    'n_estimators': 100,  # Nombre d'arbres
}
print(f"Définition des paramètres terminé en {time.time() - start_time:.2f} secondes")

# Entraîner le modèle avec GPU
print("Entraînement du modèle XGBoost avec GPU...")
start_time = time.time()
model = xgb.train(params, dtrain, num_boost_round=100)
print(f"Entraînement du modèle terminé en {time.time() - start_time:.2f} secondes")

# Prédire avec le modèle
print("Prédiction des résultats...")
start_time = time.time()
y_pred = model.predict(dtest)
print(f"Prédiction terminé en {time.time() - start_time:.2f} secondes")

# Évaluer la performance
print("Évaluation des performances du modèle...")
start_time = time.time()
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
print(f"Évaluation terminé en {time.time() - start_time:.2f} secondes")

# Sauvegarder le modèle
print("Sauvegarde du modèle...")
start_time = time.time()
joblib.dump(model, "accident_severity_model_xgboost_gpu.pkl")
print(f"Sauvegarde du modèle terminé en {time.time() - start_time:.2f} secondes")

print("Toutes les étapes sont terminées.")
metrics = {"accuracy": accuracy}

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    # Log du modèle XGBoost
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path=artifact_path,
        input_example=X_test
    )