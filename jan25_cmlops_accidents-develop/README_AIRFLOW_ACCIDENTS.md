# 📊 Airflow MLOps Pipeline – Gravité des Accidents (Datascientest 2025)  

- Ce projet définit un **DAG Airflow complet** pour l'automatisation d'un pipeline MLOps basé sur des données d'accidents de la route, avec stockage sur **MinIO**, entraînement de modèles **scikit-learn**, suivi via **MLflow**, et rechargement des modèles via une API **FastAPI**.
- La particularité de ce Pipelene airflow c'est qu'il peut prendre de façon **dynamique** n'importe quels modéle ML définie dans un dictionnaire "models" 
- l'ajout d'un nouveau modèle d'apprentissage automatique ne nécessite presque **aucune modification du DAG**( Cf Annexes au bas du document)
- Nous pouvons dans une prochaine version externaliser le dictionnaire dans un fichier json loadé par la premiére tâche du Dag.

---

## 📦 Structure du pipeline

### 1. `load_csv_to_s3`
- Upload des fichiers CSV bruts (`/app/data/raw/*.csv`) dans le bucket MinIO `accidents-raw-csv`.

### 2. `prepare_data`
- Télécharge les fichiers depuis MinIO.
- Fusionne les fichiers par année (`caracteristiques`, `lieux`, `usagers`, `vehicules`) selon `Num_Acc` et `id_vehicule`.
- Enregistre les datasets fusionnés dans `accidents-preprocessed-csv`.

### 3. `clean_data`
- Nettoie le dataset global :
  - Convertit les colonnes en types numériques.
  - Supprime les colonnes inutiles.
  - Traite la variable `nbv`.
  - Crée des splits `train_test` et `global_test`.
- Upload des jeux de données nettoyés dans MinIO.

---

## 🔍 Recherche & entraînement de modèles

Pour chaque modèle (`SVC`, `MLP`, `RandomForest`), les étapes suivantes sont exécutées :

### 4. `search_params_{model_name}`
- GridSearchCV sur le modèle pour trouver les meilleurs hyperparamètres.

### 5. `train_model_{model_name}`
- Entraîne le modèle avec les meilleurs paramètres trouvés.

### 6. `log_model_{model_name}`
- Évalue le modèle par cross-validation.
- Log dans **MLflow** avec métriques :
  - `cv_accuracy`
  - `cv_f1_weighted`
  - `cv_precision_weighted`
  - `cv_recall_weighted`

### 7. `reload_model_{model_name}`
- Envoie une requête à l'API pour recharger le modèle dans FastAPI.

---

## 🏆 Sélection du meilleur modèle

### 8. `select_best_model`
- Sélectionne le meilleur run MLflow selon une combinaison pondérée des métriques.
- Enregistre et met en production via `mlflow.register_model(...)`.

### 9. `reload_best_model`
- Requête vers FastAPI pour charger le modèle en production.

---

## 🧰 Technologies utilisées

| Composant                   | Description               |
| --------------------------- | ------------------------- |
| **Airflow**                 | Orchestration du pipeline |
| **MinIO**                   | Stockage type S3          |
| **Pandas / Scikit-learn**   | Traitement & modèles      |
| **MLflow**                  | Suivi des expériences     |
| **FastAPI**                 | Déploiement des modèles   |
| **Docker / Docker Compose** | Conteneurisation          |

---

## 🚀 Lancer le DAG

1. Assurez-vous que :
   - MinIO est accessible (`minio:9000`)
   - Airflow est configuré et fonctionnel
   - MLflow fonctionne (serveur activé)

2. Placez vos CSV dans `/app/data/raw/`.

3. Activez manuellement le DAG `ml_accidents_pipeline_dag` via l'interface Airflow.

---

## 📁 Buckets MinIO

| Bucket                       | Contenu                        |
| ---------------------------- | ------------------------------ |
| `accidents-raw-csv`          | Fichiers CSV bruts             |
| `accidents-preprocessed-csv` | Fichiers nettoyés et fusionnés |
| `accidents-train-csv`        | Splits d'entraînement et test  |
| `accidents-models`           | Modèles entraînés (`.pkl`)     |

---

## 📌 Remarques

- Les modèles sont entraînés avec des pipelines `Pipeline([Scaler, Classifier])`.
- Le `dag_id` est : `ml_accidents_pipeline_dag`.
- Le DAG n'est pas planifié (`schedule_interval=None`) — exécution manuelle recommandée.

## 🧩 Annexe – Pipeline dynamique : Ajouter facilement des modèles

Ce pipeline Airflow est **entièrement dynamique** : l'ajout d'un nouveau modèle d'apprentissage automatique ne nécessite **aucune modification du DAG**.

---

### 🛠️ Étape 1 – Ajouter un modèle dans le dictionnaire `models`

L’ensemble des modèles est défini dans le DAG principal, dans une structure Python comme celle-ci :

```python
models = {
    "RandomForest": {
        "pipeline": Pipeline([("clf", RandomForestClassifier(random_state=42))]),
        "param_grid": {"clf__n_estimators": [10], "clf__max_depth": [None]}
    },
    "MLP": {
        "pipeline": Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=300))]),
        "param_grid": {"clf__hidden_layer_sizes": [(10,)], "clf__alpha": [0.001]}
    }
}
```
- Exemple : Ajouter plusieurs nouveaux modèles
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

models["LogisticRegression"] = {
    "pipeline": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ]),
    "param_grid": {
        "clf__C": [0.1, 1.0],
        "clf__penalty": ["l2"]
    }
}

models["GradientBoosting"] = {
    "pipeline": Pipeline([
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),
    "param_grid": {
        "clf__n_estimators": [50],
        "clf__learning_rate": [0.1]
    }
}

```
- Grace à la boucle :
```python
for model_name in models.keys():
    ...
```
- Airflow crée automatiquement toutes les tâches nécessaires pour chaque modèle :
  - search_params_LogisticRegression
  - train_model_LogisticRegression
  - log_model_LogisticRegression
  - reload_model_LogisticRegression
  - Et de même pour GradientBoosting.

### Avantages :

- 🔁 Ajout instantané de modèles sans modifier les tâches.

- 🧪 Idéal pour les comparaisons ou benchmarks.

- 📈 Intégration transparente avec MLflow et FastAPI.

- 🚀 Prêt pour XGBoost, LightGBM, KNN, etc.

- **Extensibilité**      : Ajoutez autant de modèles que vous voulez.
- **Suivi automatique**  :  MLflow.
- **Déploiement simple** : via API FastAPI.

### En résumé
- Pour ajouter un modèle :
  - Ajoutez-le dans le dictionnaire models avec sa pipeline et param_grid.
  - Lancez le DAG — le pipeline s’adapte automatiquement.

