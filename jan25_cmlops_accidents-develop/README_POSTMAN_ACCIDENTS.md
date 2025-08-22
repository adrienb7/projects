# 📘 README_POSTMAN_ACCIDENTS.md

Ce document décrit comment utiliser l’API FastAPI pour la prédiction des accidents via **Postman**, en incluant :

- Les endpoints disponibles
- La sécurisation via API Key
- L'utilisation des paramètres `model_name` et `version`
- L’environnement Postman prêt à l’emploi

---

## ✅ 1. Authentification avec Clé API

Toutes les routes sécurisées nécessitent l’envoi d’un header :

```http
X-Api-Key: 2eeq-3Ic1-z4Cq-2fdg-0j62-y1B4
| Méthode | Endpoint                                       | Description                      | Sécurisé |
| ------- | ---------------------------------------------- | -------------------------------- | -------- |
| `GET`   | `/test`                                        | Vérifie que l’API fonctionne     | ❌        |
| `GET`   | `/test/{echo}`                                 | Vérifie l’écho + API Key         | ✅        |
| `POST`  | `/predict/{model_name}`                        | Prédiction avec un modèle        | ✅        |
| `POST`  | `/reload_model/{model_name}?version={version}` | Recharge un modèle depuis MLflow | ✅        |
```
## ⚙️ 3. Utilisation des Paramètres 

- 🔸 Path Parameter : model_name
-  du modèle à utiliser, ex :
  - RandomForest
  - MLP
  - SVC
  - BestModel

- 🔸 Query Parameter : version (optionnel)
  - Permet de spécifier une version précise du modèle à charger (via MLflow).
  - Si non précisé, la dernière version publiée sera utilisée automatiquement.

## 4. Exemple d'Appels API avec Postman

- Étape 1 : Recharger un modèle depuis MLflow
```http
POST http://0.0.0.0:8089/api/v1/reload_model/RandomForest
Headers:
  X-Api-Key: 2eeq-3Ic1-z4Cq-2fdg-0j62-y1B4
```

- Avec une version spécifique :
```http 
p://0.0.0.0:8089/api/v1/reload_model/RandomForest?version=2
Headers:
  X-Api-Key: 2eeq-3Ic1-z4Cq-2fdg-0j62-y1B4
```

- Étape 2 : Envoyer une prédiction
```http
POST http://0.0.0.0:8089/api/v1/predict/RandomForest
Headers:
  X-Api-Key: 2eeq-3Ic1-z4Cq-2fdg-0j62-y1B4
Body (JSON):
{
  "lum": 1,
  "agg": 1,
  "int": 1,
  "atm": 1,
  "col": 2,
  "com": 75000,
  "dep": 75,
  "an": 2022,
  "mois": 6,
  "jour": 15,
  "hrmn": "14:30",
  "lat": 48.8566,
  "long": 2.3522,
  "dep_code": "75"
}
```

## 5. Environnement Postman (à importer)
- Vous pouvez importer l’environnement suivant dans Postman (.json) pour centraliser les variables comme host, clés, etc.
- fichier : **postman/Soutenance Accidents.postman_environment.json**

## 📎 Remarques
- Les modèles sont chargés depuis MLflow Tracking Server.

- L’environnement est prévu pour tourner dans un conteneur Docker (FastAPI accessible sur http://0.0.0.0:8089).

- Assurez-vous que les ports Docker sont bien exposés en local.

- Si vous modifiez la clé API, pensez à mettre à jour l’environnement Postman également.


