import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import time
import torch

# Charger les données
print("Chargement des données...")
start_time = time.time()
df = pd.read_csv("df_accidents_nettoye_train.csv", delimiter=";", encoding="utf-8", quotechar='"', on_bad_lines='skip')
print(f"Chargement des données terminé en {time.time() - start_time:.2f} secondes")

# Séparer les caractéristiques (X) et la cible (y)
print("Séparation des caractéristiques et de la cible...")
start_time = time.time()
X = df.drop(columns=['grav'])  # Features
y = df['grav']  # Target
print(f"Séparation des caractéristiques et de la cible terminé en {time.time() - start_time:.2f} secondes")

# Convertir les variables catégorielles en numériques (si nécessaire)
print("Conversion des variables catégorielles en numériques...")
start_time = time.time()
X = pd.get_dummies(X, drop_first=True)  # Transformer les variables catégorielles en variables numériques
print(f"Conversion des variables catégorielles terminé en {time.time() - start_time:.2f} secondes")

# Vérifier les types des colonnes
print("Vérification des types des colonnes...")
start_time = time.time()
print(X.dtypes)  # Affichez les types de données des colonnes
print(f"Vérification des types terminé en {time.time() - start_time:.2f} secondes")

# Convertir toutes les colonnes en types numériques (si nécessaire)
print("Conversion des colonnes en types numériques...")
start_time = time.time()
X = X.apply(pd.to_numeric, errors='coerce')  # Convertir les colonnes en numériques, avec des NaN pour les valeurs incorrectes
print(f"Conversion des colonnes terminé en {time.time() - start_time:.2f} secondes")

# Vérifier s'il y a des NaN et les remplir
print("Remplissage des valeurs NaN...")
start_time = time.time()
X = X.fillna(X.mean())  # Remplacer les NaN par la moyenne de chaque colonne
print(f"Remplissage des NaN terminé en {time.time() - start_time:.2f} secondes")

# Diviser en ensembles d'entraînement et de test
print("Division des données en ensembles d'entraînement et de test...")
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Division des données terminé en {time.time() - start_time:.2f} secondes")

# Convertir en numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Vérifier les types avant la conversion en tensor
print("Vérification des types avant conversion en tensor...")
print("Types des données de X_train :", X_train.dtype)

# Préparer les données (assurez-vous que X_train et y_train sont des DataFrame pandas avant)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Convertir en Tensor PyTorch de type float32
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Pour classification, les labels sont des entiers
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
# Créer et initialiser le modèle TabNet
model = TabNetClassifier(
    n_d=8,  # Dimension des couches encodées
    n_a=8,  # Dimension des couches d'attention
    n_steps=3,  # Nombre de étapes de mise à jour
    gamma=1.5,  # Paramètre de régularisation
    lambda_sparse=1e-4,  # Paramètre de régularisation de la sparsité
    optimizer_fn=torch.optim.Adam,  # Optimiseur
    optimizer_params=dict(lr=2e-2),  # Paramètre du taux d'apprentissage
    mask_type='sparsemax',
    device_name='cuda'   # Utiliser sparsemax comme fonction d'attention
)

# Entraînement du modèle avec les données
model.fit(
    X_train=X_train.astype(np.float32),  # Les données d'entraînement
    y_train=y_train, 
    eval_set=[(X_test.astype(np.float32), y_test)],
    eval_name=["val"],
    eval_metric=["accuracy"], # Les étiquettes d'entraînement
    max_epochs=2,  # Nombre d'époques
    batch_size=1024,  # Taille du batch
    virtual_batch_size=128,  # Taille des sous-batches virtuels
    num_workers=0,  # Nombre de workers pour charger les données
    drop_last=False,  # Gérer les derniers batches
    loss_fn=torch.nn.CrossEntropyLoss()  # Fonction de perte pour classification multi-classes
)
print(f"Entraînement du modèle terminé en {time.time() - start_time:.2f} secondes")

# Prédictions avec le modèle
print("Prédiction des résultats...")
start_time = time.time()
y_pred = model.predict(X_test_tensor)
print(f"Prédiction terminée en {time.time() - start_time:.2f} secondes")

# Évaluation de la performance
print("Évaluation des performances du modèle...")
start_time = time.time()
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
print(f"Évaluation terminée en {time.time() - start_time:.2f} secondes")

# Sauvegarde du modèle
print("Sauvegarde du modèle...")
start_time = time.time()
joblib.dump(model, "accident_severity_model_tabnet.pkl")
print(f"Sauvegarde du modèle terminée en {time.time() - start_time:.2f} secondes")

print("Toutes les étapes sont terminées.")
