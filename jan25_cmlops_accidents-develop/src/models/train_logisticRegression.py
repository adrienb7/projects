import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

# Charger les données
print("Chargement des données...")
start_time = time.time()
df = pd.read_csv("df_accidents_sample.csv", delimiter=",", encoding="utf-8", quotechar='"', on_bad_lines='skip')
print(f"Chargement des données terminé en {time.time() - start_time:.2f} secondes")

# Supprimer les colonnes inutiles
print("Suppression des colonnes inutiles...")
start_time = time.time()
drop_columns = ['Num_Acc', 'adr', 'lat', 'long', 'voie', 'v1', 'v2', 'pr', 'pr1', 'id_vehicule', 'num_veh_x', 'num_veh_y', 'id_usager']
df = df.drop(columns=[col for col in drop_columns if col in df.columns])
print(f"Suppression des colonnes terminé en {time.time() - start_time:.2f} secondes")

# Supprimer les lignes avec des valeurs manquantes dans la colonne cible 'grav'
print("Suppression des lignes avec valeurs manquantes...")
start_time = time.time()
df = df.dropna(subset=['grav'])
print(f"Suppression des lignes terminé en {time.time() - start_time:.2f} secondes")

# Séparer les caractéristiques (X) et la cible (y)
print("Séparation des caractéristiques et de la cible...")
X = df.drop(columns=['grav'])  # Features
y = df['grav']  # Target
print(f"Séparation terminé en {time.time() - start_time:.2f} secondes")

# Convertir les colonnes catégorielles en numériques (si nécessaire)
print("Conversion des variables catégorielles en numériques...")
start_time = time.time()
X = pd.get_dummies(X, drop_first=True)
print(f"Conversion terminé en {time.time() - start_time:.2f} secondes")

# Diviser en ensembles d'entraînement et de test
print("Division des données en ensembles d'entraînement et de test...")
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Division terminé en {time.time() - start_time:.2f} secondes")

# Remplir les NaN avec la moyenne
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Entraîner le modèle Logistic Regression
print("Entraînement du modèle Logistic Regression...")
start_time = time.time()
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes")

# Prédire avec le modèle
print("Prédiction des résultats...")
start_time = time.time()
y_pred = model.predict(X_test)
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
joblib.dump(model, "accident_severity_model_lr.pkl")
print(f"Sauvegarde terminé en {time.time() - start_time:.2f} secondes")

print("Toutes les étapes sont terminées.")
