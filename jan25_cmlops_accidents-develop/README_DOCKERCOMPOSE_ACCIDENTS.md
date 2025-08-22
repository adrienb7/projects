# 📦 README — Docker Compose : Architecture Microservices (Accidents) 

Ce document décrit les services Docker utilisés dans le projet "Accidents", incluant leurs ports, réseaux, dépendances, volumes, et configuration générale.

---

## 🌐 Réseaux

| Nom         | Type    | Sous-réseau      |
|-------------|---------|------------------|
| front-tier  | bridge  | 172.22.0.0/16    |
| back-tier   | bridge  | 172.21.0.0/16    |

> `front-tier` est utilisé pour les services accessibles de l’extérieur, `back-tier` pour la communication interne.

---

## 💾 Volumes

| Nom                              | Description                       |
|----------------------------------|-----------------------------------|
| postgres_data_accidents          | Données PostgreSQL persistantes   |
| postgres_data_accidents_backup   | Sauvegardes PostgreSQL            |
| minio_data                       | Données du service MinIO          |

---

## 🔧 Services Docker

### 1. `postgresql_accidents`

> Base de données principale (PostgreSQL).

- 📦 **Ports** : `5435:${PG_PORT}`
- 🧩 **Volumes** :
  - `postgres_data_accidents:/var/lib/postgresql/data`
  - `postgres_data_accidents_backup:/var/lib/postgresql/backup`
  - `./db/conf/postgresql.conf:/etc/postgresql/postgresql.conf`
- 🔁 **Réseaux** : `back-tier`
- ❤️ **Healthcheck** : `pg_isready -U airflow`
- 📄 **Variables** : `POSTGRES_USER`, `POSTGRES_PASSWORD`

---

### 2. `adminer`

> Interface Web pour interagir avec PostgreSQL.

- 🌍 **Port** : `8088:8080`
- 🔗 **Dépendances** : `postgresql_accidents`
- 🔁 **Réseaux** : `front-tier`, `back-tier`

---

### 3. `app` (FastAPI)

> Serveur d'API pour les prédictions d'accidents.

- 📦 **Ports** : `${EXT_FAST_API_PORT}:${LOCAL_FAST_API_PORT}`
- 📂 **Volumes** : `./fastapi/app:/app`
- 🔗 **Dépendances** : `mlflow-server`, `postgresql_accidents`
- 🔁 **Réseaux** : `front-tier`, `back-tier`
- 🧪 **Healthcheck** : `GET /api/v1/test`
- 🔐 **Clé API** requise

---

### 4. `mlflow-server`

> Serveur de gestion de modèles MLflow (stockage S3 via MinIO).

- 🌍 **Ports** : `${MLFLOW_PORT}:${MLFLOW_PORT}`
- 🔗 **Dépendances** : `postgresql_accidents`, `minio`
- 📂 **Command** :
  - Migration de la base
  - Lancement de `mlflow server` avec artefacts sur MinIO
- 🔁 **Réseaux** : `front-tier`, `back-tier`
- 🧪 **Healthcheck** : `GET /`

---

### 5. `minio`

> Service de stockage compatible S3.

- 🌍 **Ports** :
  - API : `${MINIO_API_ADDRESS}:${MINIO_API_PORT}`
  - Console : `${MINIO_CONSOLE_ADDRESS}:${MINIO_CONSOLE_PORT}`
- 📂 **Volumes** : `minio_data:/data`
- 🔁 **Réseaux** : `front-tier`, `back-tier`
- 🧪 **Healthcheck** : `GET /minio/health/live`

---

### 6. `redis`

> Broker de messages utilisé par Celery (Airflow).

- 📦 **Port** : `6379:6379`
- 🔁 **Réseau** : `back-tier`
- 🧪 **Healthcheck** : `redis-cli ping`

---

### 7. `airflow-webserver`

> Interface Web d’Airflow pour la gestion des DAGs.

- 🌍 **Port** : `8080:8080`
- 🔁 **Réseaux** : `front-tier`, `back-tier`
- 🧪 **Healthcheck** : `GET /health`

---

### 8. `airflow-scheduler`

> Ordonnanceur principal de tâches Airflow.

- 🛠️ **Command** : `airflow scheduler`
- 🧪 **Healthcheck** : `airflow jobs check --job-type SchedulerJob`

---

### 9. `airflow-worker`

> Worker Celery exécutant les tâches planifiées.

- 🛠️ **Command** : `celery worker`
- 🧪 **Healthcheck** : `celery inspect ping`

---

### 10. `airflow-init`

> Initialisation de la base Airflow (création user, upgrade).

- 🛠️ **Command** : `airflow version`
- 🧬 **Variables** :
  - `_AIRFLOW_DB_UPGRADE=true`
  - `_AIRFLOW_WWW_USER_CREATE=true`

---

### 11. `flower`

> Dashboard de monitoring Celery.

- 🌍 **Port** : `5555:5555`
- 🧪 **Healthcheck** : `GET /`

---

## ✅ Récapitulatif des Ports Exposés

| Service             | Port Exposé                  |
|---------------------|------------------------------|
| `postgresql_accidents` | 5435                     |
| `adminer`           | 8088                         |
| `app (FastAPI)`     | `${EXT_FAST_API_PORT}`       |
| `mlflow-server`     | `${MLFLOW_PORT}`             |
| `minio` (API + UI)  | `${MINIO_API_PORT}`, `${MINIO_CONSOLE_PORT}` |
| `airflow-webserver` | 8080                         |
| `flower`            | 5555                         |

---

## 🔐 Sécurité & Authentification

- **Clé API (FastAPI)** : Obligatoire pour accéder aux endpoints `/predict/...` et `/reload_model/...`
- **Réseaux** :
  - `back-tier` : Sécurise la communication entre services critiques (Bdd, MLflow, MinIO)
  - `front-tier` : Permet l'accès aux interfaces Adminer, API FastAPI, etc.

---

## 📁 Volumes de données persistants

| Volume                       | Monté dans                         |
|------------------------------|------------------------------------|
| `postgres_data_accidents`    | `/var/lib/postgresql/data`         |
| `postgres_data_accidents_backup` | `/var/lib/postgresql/backup`   |
| `minio_data`                 | `/data`                            |

---

## 🧪 Healthchecks intégrés

| Service             | Test utilisé                                 |
|---------------------|----------------------------------------------|
| `postgresql_accidents` | `pg_isready`                             |
| `app`               | `GET /api/v1/test`                           |
| `mlflow-server`     | `GET /`                                      |
| `minio`             | `GET /minio/health/live`                     |
| `redis`             | `redis-cli ping`                             |
| `airflow-webserver` | `GET /health`                                |
| `airflow-scheduler` | `airflow jobs check --job-type SchedulerJob` |
| `airflow-worker`    | `celery inspect ping`                        |
| `flower`            | `GET /`                                      |

---

## 📌 Notes supplémentaires

- La configuration utilise des fichiers `.env` pour injecter les variables sensibles.
- Les services sont isolés dans des sous-réseaux pour une meilleure sécurité et performance.
- L'infrastructure est prête pour les environnements de test, développement ou démonstration.

---

