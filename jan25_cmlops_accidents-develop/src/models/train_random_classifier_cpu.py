import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Charger les données
try:
    # Lire le fichier CSV avec délimiteurs et guillemets gérés
    df = pd.read_csv("df_accidents.csv", delimiter=",", encoding="utf-8", quotechar='"')
    print("Données chargées avec succès !")
    print("Shape avant nettoyage:", df.shape)  # Afficher la forme du DataFrame avant nettoyage
    print(df.head())  # Afficher les premières lignes pour vérifier

except Exception as e:
    print("Erreur lors du chargement du fichier CSV :", e)

# Vérifier s'il y a des lignes vides
print("Vérification des lignes vides dans le DataFrame:")
print(df.isnull().sum())  # Compter les valeurs manquantes par colonne

# Supprimer les colonnes inutiles (identifiants, coordonnées, texte)
drop_columns = ['Num_Acc', 'adr', 'lat', 'long', 'voie', 'v1', 'v2', 'pr', 'pr1', 'id_vehicule', 'num_veh_x', 'num_veh_y', 'id_usager']
df = df.drop(columns=[col for col in drop_columns if col in df.columns])

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna(subset=['grav'])

# Afficher la forme du DataFrame après nettoyage
print("Shape après nettoyage:", df.shape)
print(df.head())  # Afficher à nouveau les premières lignes après nettoyage

# Vérifier si après nettoyage, le DataFrame a suffisamment de données
if df.shape[0] < 5:
    print("Le DataFrame contient moins de 5 échantillons. Impossible de faire une validation croisée avec 5 plis.")
else:
    # Séparer les caractéristiques (X) et la cible (y)
    X = df.drop(columns=['grav'])  # Features
    y = df['grav']  # Target

    # Convertir les colonnes catégorielles en numériques (si nécessaire)
    X = pd.get_dummies(X, drop_first=True)

    # Créer le modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Appliquer la validation croisée
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

    # Afficher les résultats de la validation croisée
    print(f"Scores de la validation croisée : {scores}")
    print(f"Précision moyenne : {scores.mean():.4f}")
    print(f"Écart-type des scores : {scores.std():.4f}")

    # Entraîner le modèle sur l'ensemble complet des données
    model.fit(X, y)

    # Sauvegarder le modèle
    joblib.dump(model, "accident_severity_model.pkl")
    print("Modèle entraîné et sauvegardé sous 'accident_severity_model.pkl'")

    # Optionnel : si vous voulez obtenir un rapport détaillé sur la performance
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
