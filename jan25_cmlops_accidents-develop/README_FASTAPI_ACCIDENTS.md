# README_FASTAPI_PREDICT.md 

# API FastAPI pour Prédiction ML avec MLflow (Docker)

Cette API permet de charger dynamiquement des modèles MLflow et d’effectuer des prédictions sécurisées.

---

## Sécurisation

- Tous les endpoints sensibles nécessitent une **clé API** passée dans le header HTTP.
- Nom du header : défini par `API_KEY_NAME` dans la config (exemple : `x-api-key`).
- Exemple header :  


---

## Endpoints Principaux

### 1. Test API basique  
**GET** `/test`  
- Test simple, pas de clé API requise.  
- Retourne `{ "detail": "API FONCTIONNELLE" }`

---

### 2. Test API sécurisé  
**GET** `/test/{echo}`  
- Test avec message personnalisé `echo` en path paramètre.  
- Clé API obligatoire.

---

### 3. Prédiction  
**POST** `/predict/{model_name}`  
- Fait une prédiction avec un modèle chargé.  
- `model_name` (path param) : nom du modèle MLflow.  
- Corps JSON : données conformes au modèle `PredictAccidentsPayload`.  
- Clé API obligatoire.  
- Retour : prédiction + version modèle.

---

### 4. Rechargement dynamique modèle  
**POST** `/reload_model/{model_name}`  
- Recharge un modèle MLflow dans l’API.  
- `model_name` (path param) : nom du modèle à recharger.  
- Paramètre optionnel `version` (query param) : version du modèle MLflow à charger.  
- Si `version` non fournie, la dernière version est chargée.  
- Clé API obligatoire.  
- Retour : message succès + version chargée.

---

## Exemple d’appel avec `curl`

```bash
# Recharger modèle version 2
curl -X POST "http://localhost:8000/reload_model/RandomForest?version=2" \
-H "x-api-key: CLE_API"

# Faire une prédiction
curl -X POST "http://localhost:8000/predict/RandomForest" \
-H "Content-Type: application/json" \
-H "x-api-key: CLE_API" \
-d '{ "feature1": 1.2, "feature2": 3.4, ... }'

```

## Note
- L’API dépend d’un serveur MLflow accessible depuis l’environnement Docker (mlflow-server:5000 par défaut)
- Les modèles sont identifiés par model_name et version MLflow.
- Les erreurs renvoient un code HTTP approprié (404 pour modèle non chargé, 500 pour erreur interne).